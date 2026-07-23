#!/usr/bin/env python3
"""
pacman.launch.py — Full Pac-Man multi-bot Gazebo 11 launch.

Launch order
────────────
t+0   gzserver + gzclient
t+4   robot_state_publisher  (for TF frames)
t+6   spawn_arena.py         (walls, pellets, all bots)
t+10  nrf24_bridge           (radio range-gating)
t+12  pacman_node            (pac-man AI + pellet consumption)
t+14  ghost_node × N_GHOSTS  (stub — spawned, hold position)
"""

import os
import sys
import re
import subprocess
import tempfile
import yaml
import atexit
from os import environ, pathsep

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, TimerAction, IncludeLaunchDescription
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node

# ── Resolve scripts directory at module-load time ─────────────────────────────
# This runs when ros2 loads the launch file (before generate_launch_description),
# so sys.path is ready for the maze_generator import inside the function.

def _find_scripts_dir() -> str:
    """Walk candidate install/source locations to find maze_generator.py."""
    # __file__ may be the installed copy: .../share/minibot/launch/pacman.launch.py
    # maze_generator lives at:           .../share/minibot/scripts/maze_generator.py
    #                        OR          .../lib/minibot/maze_generator.py
    candidates = []

    try:
        pkg = get_package_share_directory('minibot')
        candidates.append(os.path.join(pkg, 'scripts'))                        # share/minibot/scripts
        candidates.append(os.path.join(os.path.dirname(pkg),                   # lib/minibot
                                        '..', 'lib', 'minibot'))
    except Exception:
        pass

    # Relative to this file (works in both source and symlink-install layouts)
    _here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(_here, '..', 'scripts'))                    # source tree
    candidates.append(_here)

    for c in candidates:
        c = os.path.realpath(c)
        if os.path.isfile(os.path.join(c, 'maze_generator.py')):
            return c

    raise RuntimeError(
        'Cannot locate maze_generator.py — searched: ' + str(candidates))


_SCRIPTS_DIR = _find_scripts_dir()
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Now safe to import
from maze_generator import N_GHOSTS, GHOST_NAMES  # noqa: E402


def generate_launch_description():
    pkg        = get_package_share_directory('minibot')
    xacro_file = os.path.join(pkg, 'urdf', 'minibot_urdf.xacro')

    # ── Strip XML comments from URDF (avoids rcl colon-parse bug) ─────────
    urdf_xml = subprocess.check_output(['xacro', xacro_file]).decode('utf-8')
    urdf_xml = re.sub(r'<!--.*?-->', '', urdf_xml, flags=re.DOTALL)

    _tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='_robot_desc.yaml', delete=False)
    yaml.dump(
        {'robot_state_publisher': {'ros__parameters': {
            'robot_description': urdf_xml,
            'use_sim_time': True,
        }}},
        _tmp, default_flow_style=False, allow_unicode=True,
    )
    _tmp.flush()
    _tmp.close()
    params_file = _tmp.name
    atexit.register(
        lambda: os.unlink(params_file) if os.path.exists(params_file) else None)

    # ── Gazebo environment ─────────────────────────────────────────────────
    share_parent = os.path.dirname(pkg)
    model_path   = share_parent
    if 'GAZEBO_MODEL_PATH' in environ:
        model_path += pathsep + environ['GAZEBO_MODEL_PATH']

    gz_env = {
        'GAZEBO_MODEL_PATH':         model_path,
        'GAZEBO_PLUGIN_PATH':        environ.get('GAZEBO_PLUGIN_PATH', ''),
        'GAZEBO_RESOURCE_PATH':      model_path,
        'GAZEBO_MODEL_DATABASE_URI': '',
        'GAZEBO_IP':         '127.0.0.1',
        'GAZEBO_MASTER_URI': 'http://127.0.0.1:11345',
        'LIBGL_ALWAYS_SOFTWARE': '1',
        'OGRE_RTT_MODE': 'Copy',
    }

    # PYTHONPATH for every child process so they can import maze_generator
    import random
    seed_val = str(random.randint(1, 9999999))
    py_path = _SCRIPTS_DIR + pathsep + environ.get('PYTHONPATH', '')
    child_env = {**gz_env, 'PYTHONPATH': py_path, 'PACMAN_SEED': seed_val}

    # ── Arguments ──────────────────────────────────────────────────────────
    paused_arg = DeclareLaunchArgument(
        'paused', default_value='false',
        description='Start Gazebo paused')

    # ── Gazebo (via official gazebo_ros launch) ────────────────────────────
    # Setting GAZEBO_MODEL_PATH in environ so the included launch file picks it up
    os.environ['GAZEBO_MODEL_PATH'] = gz_env['GAZEBO_MODEL_PATH']
    
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')
    world = os.path.join(pkg, 'worlds', 'pacman_empty.world')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world,
            'pause': LaunchConfiguration('paused'),
        }.items()
    )

    # ── robot_state_publisher ──────────────────────────────────────────────
    rsp = TimerAction(period=4.0, actions=[
        Node(package='robot_state_publisher',
             executable='robot_state_publisher',
             name='robot_state_publisher',
             output='screen',
             parameters=[params_file])
    ])

    # ── Arena spawner (walls + pellets + bots) ─────────────────────────────
    spawn_arena = TimerAction(period=15.0, actions=[
        ExecuteProcess(
            cmd=['python3',
                 os.path.join(_SCRIPTS_DIR, 'spawn_arena.py')],
            output='screen',
            additional_env=child_env,
        )
    ])

    # ── NRF24 bridge ───────────────────────────────────────────────────────
    nrf24_bridge = TimerAction(period=19.0, actions=[
        ExecuteProcess(
            cmd=['python3',
                 os.path.join(_SCRIPTS_DIR, 'nrf24_bridge.py')],
            output='screen',
            additional_env=child_env,
        )
    ])

    # ── pacman_node ────────────────────────────────────────────────────────
    pacman_node = TimerAction(period=21.0, actions=[
        ExecuteProcess(
            cmd=['python3',
                 os.path.join(_SCRIPTS_DIR, 'pacman_node.py')],
            output='screen',
            additional_env=child_env,
        )
    ])

    # ── ghost_nodes (one process per ghost, staggered by 0.5 s) ───────────
    ghost_nodes = []
    for i, gname in enumerate(GHOST_NAMES):
        ghost_nodes.append(TimerAction(
            period=23.0 + i * 0.5,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'python3',
                        os.path.join(_SCRIPTS_DIR, 'ghost_node.py'),
                        '--ros-args',
                        '-p', f'ghost_id:={i}',
                        '-p', f'ghost_name:={gname}',
                    ],
                    output='screen',
                    additional_env=child_env,
                )
            ]
        ))

    # ── Pygame Visualizer ──────────────────────────────────────────────────
    visualizer = TimerAction(period=25.0, actions=[
        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(_SCRIPTS_DIR, 'ros_visualizer.py')
            ],
            output='screen',
            additional_env=child_env,
        )
    ])

    # ── Score Visualizer (dedicated terminal window) ───────────────────────
    # Try xterm first (most universally available), fall back to
    # gnome-terminal / x-terminal-emulator if xterm is missing.
    score_vis_cmd = [
        'bash', '-c',
        (
            'TITLE="PAC-MAN STATS"; '
            f'PYPATH="{py_path}"; '
            'CMD="python3 {script} 2>&1"; '
            'if command -v xterm &>/dev/null; then '
            f'    xterm -title "$TITLE" -bg black -fg green -geometry 72x28 '
            f'    -e "PYTHONPATH={py_path} python3 {os.path.join(_SCRIPTS_DIR, "score_visualizer.py")} 2>&1"; '
            'elif command -v gnome-terminal &>/dev/null; then '
            f'    gnome-terminal --title "$TITLE" -- bash -c '
            f'    "PYTHONPATH={py_path} python3 {os.path.join(_SCRIPTS_DIR, "score_visualizer.py")} 2>&1; read"; '
            'else '
            f'    x-terminal-emulator -e "PYTHONPATH={py_path} python3 {os.path.join(_SCRIPTS_DIR, "score_visualizer.py")} 2>&1"; '
            'fi'
        )
    ]

    score_visualizer = TimerAction(period=27.0, actions=[
        ExecuteProcess(
            cmd=score_vis_cmd,
            output='screen',
            additional_env=child_env,
        )
    ])

    return LaunchDescription([
        paused_arg,
        gazebo,
        rsp,
        spawn_arena,
        nrf24_bridge,
        pacman_node,
        *ghost_nodes,
        visualizer,
        score_visualizer,
    ])
