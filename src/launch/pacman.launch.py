#!/usr/bin/env python3
"""
pacman.launch.py — Pac-Man Gazebo arena.

Generates a random maze, spawns it as a Gazebo world, then places
8 colour-coded minibots (1 pacman + 7 ghosts) each in its own
racetrack channel so they never collide.

Usage:
  ros2 launch minibot pacman.launch.py

Each bot is namespaced (e.g. /pacman/cmd_vel, /ghost_0/cmd_vel) and
controlled via the planar_move Gazebo plugin.  A separate game-controller
node (future) will drive them with the pacmanbot AI.
"""
import os
import re
import json
import subprocess
import tempfile
import yaml
import atexit
from os import environ, pathsep

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


# ── Bot definitions (colours from pacman.py) ──────────────────────────
BOT_DEFS = [
    # (name,      R/255,  G/255,  B/255)  — normalised for xacro
    ('pacman',    1.000, 0.863, 0.000),   # yellow
    ('ghost_0',   0.863, 0.118, 0.118),   # red
    ('ghost_1',   1.000, 0.392, 0.706),   # pink
    ('ghost_2',   0.000, 0.863, 0.863),   # cyan
    ('ghost_3',   1.000, 0.627, 0.118),   # orange
    ('ghost_4',   0.706, 0.000, 0.706),   # purple
    ('ghost_5',   0.000, 0.706, 0.314),   # green
    ('ghost_6',   1.000, 1.000, 1.000),   # white
]


def generate_launch_description():
    pkg = get_package_share_directory('minibot')
    xacro_file = os.path.join(pkg, 'urdf', 'minibot_game.xacro')

    # ── 1. Generate the maze world ────────────────────────────────────
    gen_script = os.path.join(pkg, 'scripts', 'generate_maze.py')
    # Ensure the pacmanbot source is findable
    pacmanbot_dir = os.path.join(pkg, 'pacmanbot')
    env_with_path = dict(environ)
    env_with_path['PYTHONPATH'] = pacmanbot_dir + pathsep + env_with_path.get('PYTHONPATH', '')

    raw = subprocess.check_output(
        ['python3', gen_script],
        env=env_with_path,
        stderr=subprocess.PIPE,
    ).decode('utf-8').strip()

    maze_info    = json.loads(raw)
    world_path   = maze_info['world_path']
    spawns       = maze_info['spawns']          # {bot_name: {x, y}}
    grid_json    = json.dumps(maze_info['grid'])
    player_start = '{},{}'.format(*maze_info['player_start'])
    cell_pitch   = float(maze_info['cell_pitch'])

    # Clean up the world file on exit
    atexit.register(lambda: os.unlink(world_path) if os.path.exists(world_path) else None)

    # ── 2. Gazebo environment (same fixes as gazebo.launch.py) ────────
    share_parent = os.path.dirname(pkg)

    model_path = share_parent
    if 'GAZEBO_MODEL_PATH' in environ:
        model_path += pathsep + environ['GAZEBO_MODEL_PATH']

    plugin_path = environ.get('GAZEBO_PLUGIN_PATH', '')
    media_path = share_parent
    if 'GAZEBO_RESOURCE_PATH' in environ:
        media_path += pathsep + environ['GAZEBO_RESOURCE_PATH']

    gz_env = {
        'GAZEBO_MODEL_PATH':        model_path,
        'GAZEBO_PLUGIN_PATH':       plugin_path,
        'GAZEBO_RESOURCE_PATH':     media_path,
        'GAZEBO_MODEL_DATABASE_URI': '',
        'GAZEBO_IP':                '127.0.0.1',
        'GAZEBO_MASTER_URI':        'http://127.0.0.1:11345',
        'LIBGL_ALWAYS_SOFTWARE':    '1',
        'OGRE_RTT_MODE':           'Copy',
    }

    # ── 3. gzserver + gzclient ────────────────────────────────────────
    gzserver = ExecuteProcess(
        cmd=[
            'gzserver', world_path,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
            '-s', 'libgazebo_ros_state.so',
        ],
        output='screen',
        additional_env=gz_env,
    )

    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
        additional_env=gz_env,
    )

    # ── 4. Per-bot: xacro → URDF → RSP + spawn ───────────────────────
    bot_actions = []
    for bot_name, br, bg, bb in BOT_DEFS:
        # Expand xacro with colour args
        urdf_xml = subprocess.check_output([
            'xacro', xacro_file,
            f'robot_name:={bot_name}',
            f'body_r:={br}',
            f'body_g:={bg}',
            f'body_b:={bb}',
        ]).decode('utf-8')
        # Strip XML comments (same rcl parser workaround as gazebo.launch.py)
        urdf_xml = re.sub(r'<!--.*?-->', '', urdf_xml, flags=re.DOTALL)

        # Write to temp YAML params file
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix=f'_{bot_name}_desc.yaml', delete=False)
        yaml.dump(
            {'/**': {'ros__parameters': {
                'robot_description': urdf_xml,
                'use_sim_time': True,
                'frame_prefix': f'{bot_name}/',
            }}},
            tmp, default_flow_style=False, allow_unicode=True,
        )
        tmp.flush()
        tmp.close()
        params_file = tmp.name
        atexit.register(lambda p=params_file: os.unlink(p) if os.path.exists(p) else None)

        # robot_state_publisher (namespaced)
        rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=bot_name,
            output='screen',
            parameters=[params_file],
        )
        bot_actions.append(rsp)

        # Spawn entity (delayed 4s for gzserver to settle)
        sp = spawns[bot_name]
        spawn = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name=f'spawn_{bot_name}',
            output='screen',
            arguments=[
                '-topic', f'/{bot_name}/robot_description',
                '-entity', bot_name,
                '-x', str(sp['x']),
                '-y', str(sp['y']),
                '-z', '0.035',
            ],
        )
        bot_actions.append(spawn)

    # Delay bot spawning to let Gazebo start
    delayed_bots = TimerAction(period=4.0, actions=bot_actions)

    # ── 5. Game controller node (delayed 8 s — after bots have spawned) ───
    game_node = Node(
        package='minibot',
        executable='pacman_node.py',
        name='pacman_game_node',
        output='screen',
        parameters=[{
            'grid_json':    grid_json,
            'player_start': player_start,
            'cell_pitch':   cell_pitch,
            'speed':        3.0,
            'use_sim_time': True,
        }],
    )

    # ── 6. Game monitor in its own xterm window ───────────────────────
    # Derive install/setup.bash: pkg = .../install/minibot/share/minibot
    install_setup = os.path.normpath(os.path.join(pkg, '..', '..', '..', 'setup.bash'))
    monitor_cmd = (
        f'source /opt/ros/humble/setup.bash && '
        f'source {install_setup} && '
        f'ros2 run minibot game_monitor.py; '
        f'echo ""; echo "Session ended. Press Enter to close."; read'
    )
    monitor_proc = ExecuteProcess(
        cmd=[
            'xterm',
            '-title', 'Pac-Man  ┃  Game Monitor',
            '-fa',    'Monospace',
            '-fs',    '13',
            '-bg',    '#050514',
            '-fg',    '#e0e0ff',
            '-geometry', '38x14',
            '-e', 'bash', '-c', monitor_cmd,
        ],
        output='screen',
    )

    delayed_game = TimerAction(period=8.0, actions=[game_node, monitor_proc])

    return LaunchDescription([
        gzserver,
        gzclient,
        delayed_bots,
        delayed_game,
    ])
