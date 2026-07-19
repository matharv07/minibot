#!/usr/bin/env python3
"""
spawn_arena.py — Spawns all Pac-Man game elements into Gazebo Classic.

Spawns:
  1. Maze walls            (thin box SDF, named wall_<r>_<c>)
  2. Pellets               (small white sphere, named pellet_<r>_<c>)
  3. Power pellets         (larger yellow sphere, named power_<r>_<c>)
  4. Pacman bot            (minibot_game.xacro, yellow, named pacman_bot)
  5. Ghost bots x N_GHOSTS (minibot_game.xacro, ghost colors, named ghost_0..N)

IMPORTANT: In Gazebo Classic SpawnEntity, the world pose comes from
           `initial_pose` in the request — NOT from <pose> inside the SDF.
           The SDF <pose> is a local/relative offset and must stay at 0 0 0.

All SpawnEntity calls are fired as async futures simultaneously and polled
until every one resolves — the full arena appears in one pass.
"""

import os
import sys
import re
import subprocess
import time

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from maze_generator import (
    ROWS, COLS, CELL_SIZE, SPAWN_Z, PELLET_Z, POWER_Z,
    WALL, EMPTY, PELLET, POWER,
    PACMAN_NAME, GHOST_NAMES, N_GHOSTS,
    PACMAN_COLOR_RGB, GHOST_COLORS_RGB,
    grid_to_world, generate_map, compute_ghost_starts,
    pellet_positions, wall_positions,
)


# ── Wall visual dimensions ─────────────────────────────────────────────────────
WALL_H     = 0.15                  # wall height (m)
WALL_THICK = CELL_SIZE + 0.002     # full cell width + tiny overlap to seal seams

# Pellet sphere radii
PELLET_R = 0.018
POWER_R  = 0.030


# ─────────────────────────────────────────────────────────────────────────────
# Pose helper
# ─────────────────────────────────────────────────────────────────────────────

def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Pose:
    """Build a Pose message.  This is the world placement passed to SpawnEntity."""
    p = Pose()
    p.position.x = float(x)
    p.position.y = float(y)
    p.position.z = float(z)
    p.orientation.w = 1.0
    return p


# ─────────────────────────────────────────────────────────────────────────────
# SDF helpers
# RULE: <pose> inside the SDF is always "0 0 0 0 0 0" (model-local origin).
#       The actual world position is supplied via initial_pose in the request.
# ─────────────────────────────────────────────────────────────────────────────

def _box_sdf(name: str,
             sx: float, sy: float, sz: float,
             r: float, g: float, b: float, a: float = 1.0,
             is_static: bool = True) -> str:
    """SDF for a box (wall or floor).  Pose is always identity here."""
    static_tag = '<static>true</static>' if is_static else '<static>false</static>'
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    {static_tag}
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <collision name="col">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      </collision>
      <visual name="vis">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def _sphere_sdf(name: str,
                radius: float,
                r: float, g: float, b: float) -> str:
    """SDF for a gravity-free hovering pellet sphere.  Pose is always identity here."""
    er, eg, eb = r * 0.65, g * 0.65, b * 0.65
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <gravity>false</gravity>
      <inertial><mass>0.001</mass></inertial>
      <collision name="col">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <surface><contact><collide_without_contact>true</collide_without_contact></contact></surface>
      </collision>
      <visual name="vis">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <emissive>{er:.3f} {eg:.3f} {eb:.3f} 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def _robot_urdf(name: str,
                cr: float, cg: float, cb: float,
                xacro_path: str) -> str:
    """Expand minibot_game.xacro → URDF string.
    For URDF, SpawnEntity world pose is also taken from initial_pose in the request."""
    xml = subprocess.check_output([
        'xacro', xacro_path,
        f'robot_name:={name}',
        f'body_r:={cr}', f'body_g:={cg}', f'body_b:={cb}',
    ]).decode()
    xml = re.sub(r'<!--.*?-->', '', xml, flags=re.DOTALL)
    return xml


# ─────────────────────────────────────────────────────────────────────────────
# ArenaSpawner
# ─────────────────────────────────────────────────────────────────────────────

class ArenaSpawner(Node):
    """
    Builds every (name, xml, pose) triple up-front, then fires ALL
    SpawnEntity calls as async futures simultaneously.  Polls until done.
    The entire arena appears in Gazebo in one shot.
    """

    def __init__(self, grid, player_start, ghost_starts, xacro_path):
        super().__init__('arena_spawner')
        self._grid         = grid
        self._player_start = player_start
        self._ghost_starts = ghost_starts
        self._xacro_path   = xacro_path
        self._spawn_client = self.create_client(SpawnEntity, '/spawn_entity')

        self.get_logger().info('arena_spawner: waiting for /spawn_entity …')
        while not self._spawn_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('  …still waiting')

        self.get_logger().info('arena_spawner: building entity list …')
        entities = self._build_entity_list()
        self.get_logger().info(
            f'arena_spawner: firing {len(entities)} spawn requests simultaneously …')
        self._batch_spawn(entities)
        self.get_logger().info('arena_spawner: done.')

    # ── Entity builders ───────────────────────────────────────────────────────

    def _build_entity_list(self):
        """Return list of (name, xml_str, Pose) for every entity."""
        entities = []
        entities += self._grid_entities()
        entities += self._pellet_entities()
        entities += self._bot_entities()
        return entities

    def _grid_entities(self):
        """
        Combine the floor and all walls into a SINGLE static model with one link
        and many visual/collision tags. This instantly spawns the entire map grid
        in one API call, rather than 500+ calls.
        """
        rows, cols = self._grid.shape
        w = cols * CELL_SIZE + 2.0
        h = rows * CELL_SIZE + 2.0
        
        # Start SDF
        xml = [
            '<?xml version="1.0"?>',
            '<sdf version="1.6">',
            '  <model name="arena_grid">',
            '    <static>true</static>',
            '    <pose>0 0 0 0 0 0</pose>',
            '    <link name="link">'
        ]
        
        # 1. Floor
        fz = 0.01  # Box is 0.02 thick, surface at 0
        xml.append(f"""
      <collision name="floor_col">
        <pose>0 0 {fz} 0 0 0</pose>
        <geometry><box><size>{w} {h} 0.02</size></box></geometry>
      </collision>
      <visual name="floor_vis">
        <pose>0 0 {fz} 0 0 0</pose>
        <geometry><box><size>{w} {h} 0.02</size></box></geometry>
        <material>
          <ambient>0.05 0.05 0.05 1</ambient>
          <diffuse>0.05 0.05 0.05 1</diffuse>
        </material>
      </visual>""")

        # 2. Walls
        for r, c in wall_positions(self._grid):
            wx, wy, _ = grid_to_world(r, c)
            wz = WALL_H / 2.0
            idx = f"{r}_{c}"
            xml.append(f"""
      <collision name="wall_{idx}_col">
        <pose>{wx:.6f} {wy:.6f} {wz:.6f} 0 0 0</pose>
        <geometry><box><size>{WALL_THICK} {WALL_THICK} {WALL_H}</size></box></geometry>
      </collision>
      <visual name="wall_{idx}_vis">
        <pose>{wx:.6f} {wy:.6f} {wz:.6f} 0 0 0</pose>
        <geometry><box><size>{WALL_THICK} {WALL_THICK} {WALL_H}</size></box></geometry>
        <material>
          <ambient>0.04 0.04 0.60 1</ambient>
          <diffuse>0.04 0.04 0.60 1</diffuse>
        </material>
      </visual>""")

        xml.append('    </link>')
        xml.append('  </model>')
        xml.append('</sdf>')
        
        return [('arena_grid', "\n".join(xml), _pose(0.0, 0.0, 0.0))]

    def _pellet_entities(self):
        """
        Pack ALL pellets and power-pellets into ONE combined static-free SDF model
        ("pellet_field").  Each pellet is a separate named link within the model
        so SetEntityState can teleport individual ones below ground on consumption.

        One SpawnEntity call instead of 300+  →  instant load.
        """
        rows, cols = self._grid.shape
        positions  = pellet_positions(self._grid)
        if not positions:
            return []

        xml = [
            '<?xml version="1.0"?>',
            '<sdf version="1.6">',
            '  <model name="pellet_field">',
            '    <static>false</static>',
            '    <pose>0 0 0 0 0 0</pose>',
        ]

        for r, c, ct in positions:
            wx, wy, _ = grid_to_world(r, c)
            if ct == PELLET:
                name   = f'pellet_{r}_{c}'
                radius = PELLET_R
                z      = PELLET_Z
                cr, cg, cb = 1.0, 1.0, 1.0    # white
            else:
                name   = f'power_{r}_{c}'
                radius = POWER_R
                z      = POWER_Z
                cr, cg, cb = 1.0, 0.84, 0.0   # gold

            er = cr * 0.65
            eg = cg * 0.65
            eb = cb * 0.65

            xml.append(f"""
    <link name="{name}">
      <pose>{wx:.6f} {wy:.6f} {z:.6f} 0 0 0</pose>
      <gravity>false</gravity>
      <inertial><mass>0.001</mass></inertial>
      <collision name="col">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <surface><contact><collide_without_contact>true</collide_without_contact></contact></surface>
      </collision>
      <visual name="vis">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <material>
          <ambient>{cr} {cg} {cb} 1</ambient>
          <diffuse>{cr} {cg} {cb} 1</diffuse>
          <emissive>{er:.3f} {eg:.3f} {eb:.3f} 1</emissive>
        </material>
      </visual>
    </link>""")

        xml.append('  </model>')
        xml.append('</sdf>')

        sdf = '\n'.join(xml)
        return [('pellet_field', sdf, _pose(0.0, 0.0, 0.0))]

    def _bot_entities(self):
        entities = []
        # Pacman
        pr, pc = self._player_start
        px, py, _ = grid_to_world(pr, pc)
        urdf = _robot_urdf(PACMAN_NAME, *PACMAN_COLOR_RGB, self._xacro_path)
        entities.append((PACMAN_NAME, urdf, _pose(px, py, SPAWN_Z)))

        # Ghosts — each at their own unique grid position
        for i, (gr, gc) in enumerate(self._ghost_starts):
            gx, gy, _ = grid_to_world(gr, gc)
            gname = GHOST_NAMES[i]
            gcol  = GHOST_COLORS_RGB[i % len(GHOST_COLORS_RGB)]
            urdf  = _robot_urdf(gname, *gcol, self._xacro_path)
            entities.append((gname, urdf, _pose(gx, gy, SPAWN_Z)))
        return entities

    # ── Chunked async spawner ─────────────────────────────────────────────────
    #
    # Gazebo Classic's SpawnEntity service is single-threaded internally.
    # Firing 500+ futures in one go without spinning overflows the DDS service
    # queue and silently drops most requests — that's why walls disappeared.
    #
    # Fix: send CHUNK_SIZE requests, spin until that chunk fully resolves,
    # then send the next chunk.  Fast (parallel within each chunk) and safe.

    CHUNK_SIZE = 20   # simultaneous requests per batch (fewer total entities now)

    def _batch_spawn(self, entities, timeout_per_entity: float = 8.0):
        total   = len(entities)
        success = 0
        failed  = []

        for chunk_start in range(0, total, self.CHUNK_SIZE):
            chunk = entities[chunk_start : chunk_start + self.CHUNK_SIZE]

            chunk_pending = {}
            for name, xml, pose in chunk:
                req = SpawnEntity.Request()
                req.name             = name
                req.xml              = xml
                req.initial_pose     = pose
                req.robot_namespace  = ''
                chunk_pending[name]  = self._spawn_client.call_async(req)

            deadline = time.monotonic() + timeout_per_entity * len(chunk)
            while chunk_pending and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.02)
                done = [n for n, f in chunk_pending.items() if f.done()]
                for n in done:
                    res = chunk_pending.pop(n).result()
                    if res and res.success:
                        success += 1
                        self.get_logger().debug(f'  ✓ {n}')
                    else:
                        msg = res.status_message if res else 'no response'
                        failed.append(n)
                        self.get_logger().warn(f'  ✗ {n}: {msg}')

            for n in chunk_pending:
                failed.append(n)
                self.get_logger().warn(f'  ✗ {n}: timed out')

            pct = int(100 * (chunk_start + len(chunk)) / total)
            self.get_logger().info(
                f'  … {chunk_start + len(chunk)}/{total} ({pct}%)')

        self.get_logger().info(
            f'Spawn complete: {success}/{total} ok'
            + (f', {len(failed)} failed: {failed[:5]}' if failed else ''))


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)

    from ament_index_python.packages import get_package_share_directory
    pkg        = get_package_share_directory('minibot')
    xacro_path = os.path.join(pkg, 'urdf', 'minibot_game.xacro')

    seed_val = int(os.environ.get('PACMAN_SEED', 42))
    grid, player_start = generate_map(seed=seed_val)
    ghost_starts       = compute_ghost_starts(grid, player_start, N_GHOSTS)

    node = ArenaSpawner(grid, player_start, ghost_starts, xacro_path)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
