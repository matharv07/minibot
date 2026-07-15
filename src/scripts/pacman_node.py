#!/usr/bin/env python3
"""
pacman_node.py — Pac-Man bot ROS 2 node for Gazebo 11 Pac-Man simulation.

Navigation
──────────
• Follows grid centre-lines at all times.
• Uses the Adam-optimiser momentum algorithm from pacman.py to navigate.
• Receives ghost positions via NRF24 /rx topic → uses for potential-field
  repulsion/attraction (powered mode).
• When two bots share a cell, they negotiate a sideways-yield:
  the pacman bot briefly offsets ±CELL_SIZE/4 m in the perpendicular axis
  to let the other bot pass, then snaps back to centre.

Communication
─────────────
• NRF24 inbound:  /nrf24/pacman_bot/rx   (std_msgs/String, JSON)
  → ghost heartbeats give ghost (row, col) positions
• Publishes location: /pacman_bot/location  (geometry_msgs/Point, row/col)
• Publishes power:    /pacman_bot/power_state (std_msgs/Bool)

Pellets
───────
• Gazebo models named pellet_<row>_<col> / power_<row>_<col>
• Consumed by calling /gazebo/set_entity_state to teleport them to z=-1.0
  (below ground, instant — avoids DeleteEntity queue lag).

Motion
──────
• Publishes /<name>/cmd_vel (geometry_msgs/Twist) via planar_move plugin.
• Target speed: BOT_SPEED m/s along the active axis; cross-track P-gain
  snaps bot to centre-line of the other axis.
"""

import json
import math
import sys
import os
import random
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from maze_generator import (
    ROWS, COLS, CELL_SIZE, SPAWN_Z,
    WALL, EMPTY, PELLET, POWER,
    PACMAN_NAME, GHOST_NAMES, N_GHOSTS,
    NRF_RADIUS_M,
    grid_to_world, world_to_grid, cell_center_world,
    generate_map, compute_ghost_starts,
)

import pacman
from pacman import Player

# ── Motion constants ──────────────────────────────────────────────────────────
BOT_SPEED      = 0.18        # m/s along active axis (≈ safe centre-line speed)
CROSS_KP       = 3.5         # P-gain for cross-track correction
CELL_SNAP_DIST = 0.03        # m — consider bot "on cell centre" when within this
YIELD_OFFSET   = CELL_SIZE * 0.28   # m sideways offset during yield manoeuvre
YIELD_FRAMES   = 8           # control frames (~0.8 s) for yield duration

# ── Adam optimiser constants (mirrors pacman.py Player) ──────────────────────
BETA1 = 0.9
BETA2 = 0.999
EPS   = 1e-8

# ── Power pellet ──────────────────────────────────────────────────────────────
POWER_TICKS   = 80          # control-loop ticks powered stays active (80 × 0.1 s = 8 s)
PELLET_RADIUS = CELL_SIZE * 0.55   # world distance to trigger consumption

# ── NRF heartbeat ─────────────────────────────────────────────────────────────
HEARTBEAT_EVERY   = 5        # publish own location every N ticks via NRF24
GHOST_TIMEOUT     = 60       # ticks before a ghost position is considered stale


class PacmanNode(Node):

    def __init__(self):
        super().__init__('pacman_node')

        self._cbg = ReentrantCallbackGroup()

        # ── Generate map ─────────────────────────────────────────────────────
        self._grid, self._start = generate_map(seed=42)
        self._rows = self._grid.shape[0]
        self._cols = self._grid.shape[1]

        # ── Pacman game state ─────────────────────────────────────────────────
        pacman.AUTO_MODE = True
        self._player = Player(self._grid, self._start)
        
        self._row, self._col = self._start
        self._x, self._y = cell_center_world(self._row, self._col)
        self._score       = 0
        self._powered     = False
        self._dead        = False
        self._dead_timer  = 0

        # ── Ghost knowledge (from NRF24) ──────────────────────────────────────
        # gid → (row, col) | None
        self._ghost_pos:  dict[int, tuple | None] = {i: None for i in range(N_GHOSTS)}
        self._ghost_last_seen: dict[int, int]     = {}   # gid → tick

        # ── Movement / yield ─────────────────────────────────────────────────
        self._target_row  = self._row
        self._target_col  = self._col
        self._yield_left  = 0        # frames remaining in yield manoeuvre
        self._yield_axis  = None     # 'x' or 'y' — perpendicular to travel
        self._yield_sign  = 1        # +1 or -1 offset direction

        # ── NRF24 dedup ──────────────────────────────────────────────────────
        self._seen_ids: set = set()
        self._nrf_seq  = 0
        self._tick     = 0

        # ── Publishers ───────────────────────────────────────────────────────
        self._cmd_pub     = self.create_publisher(Twist,  f'/{PACMAN_NAME}/cmd_vel', 10)
        self._nrf_pub     = self.create_publisher(String, f'/nrf24/{PACMAN_NAME}/tx', 20)
        self._loc_pub     = self.create_publisher(Point,  '/pacman_bot/location',    10)
        self._power_pub   = self.create_publisher(Bool,   '/pacman_bot/power_state', 10)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Odometry, f'/{PACMAN_NAME}/odom',
                                 self._odom_cb, 10,
                                 callback_group=self._cbg)
        self.create_subscription(String, f'/nrf24/{PACMAN_NAME}/rx',
                                 self._nrf_rx_cb, 20,
                                 callback_group=self._cbg)

        # ── Gazebo SetEntityState service (pellet consumption) ────────────────
        self._set_state = self.create_client(SetEntityState, '/gazebo/set_entity_state')

        # ── Control timer: 10 Hz ─────────────────────────────────────────────
        self.create_timer(0.1, self._control_loop, callback_group=self._cbg)

        self.get_logger().info('pacman_node started')

    # ─────────────────────────────────────────────────────────────────────────
    # Subscribers
    # ─────────────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._row, self._col = world_to_grid(self._x, self._y)

    def _nrf_rx_cb(self, msg: String):
        try:
            pkt = json.loads(msg.data)
        except Exception:
            return
        mid = str(pkt.get('id', ''))
        if mid in self._seen_ids:
            return
        self._seen_ids.add(mid)
        if len(self._seen_ids) > 500:
            for k in list(self._seen_ids)[:250]:
                self._seen_ids.discard(k)

        for diff in pkt.get('diffs', []):
            if not diff:
                continue
            dtype = diff[0]
            if dtype == 'heartbeat':
                # ('heartbeat', gid, row, col, frame)
                _, gid, r, c, _ = diff
                if 0 <= gid < N_GHOSTS:
                    self._ghost_pos[gid]       = (int(r), int(c))
                    self._ghost_last_seen[gid] = self._tick
            elif dtype == 'agent':
                # ('agent', gid, row, col)
                _, gid, r, c = diff
                if 0 <= gid < N_GHOSTS:
                    self._ghost_pos[gid]       = (int(r), int(c))
                    self._ghost_last_seen[gid] = self._tick

    # ─────────────────────────────────────────────────────────────────────────
    # Pellet consumption
    # ─────────────────────────────────────────────────────────────────────────

    def _teleport_pellet(self, row: int, col: int, cell_type: int):
        model_prefix = 'power' if cell_type == POWER else 'pellet'
        model_name   = f'{model_prefix}_{row}_{col}'
        if self._set_state.service_is_ready():
            req = SetEntityState.Request()
            state = EntityState()
            state.name = model_name
            state.pose.position.x = 0.0
            state.pose.position.y = 0.0
            state.pose.position.z = -2.0   # below ground, invisible
            state.pose.orientation.w = 1.0
            req.state = state
            self._set_state.call_async(req)



    # ─────────────────────────────────────────────────────────────────────────
    # Sideways-yield collision avoidance
    # ─────────────────────────────────────────────────────────────────────────

    def _check_ghost_same_cell(self) -> bool:
        """Return True if any ghost is known to be on our current cell."""
        for gid, pos in self._ghost_pos.items():
            if pos is None:
                continue
            age = self._tick - self._ghost_last_seen.get(gid, 0)
            if age > 20:
                continue
            if pos == (self._row, self._col):
                return True
        return False

    def _start_yield(self):
        """Begin sideways-yield manoeuvre — offset perpendicular to travel."""
        dr = self._target_row - self._row
        dc = self._target_col - self._col
        # perpendicular axis: if moving along row (dc!=0) yield in y; else in x
        if dc != 0:   # moving left/right → yield in Y (row axis)
            self._yield_axis = 'y'
        else:         # moving up/down → yield in X (col axis)
            self._yield_axis = 'x'
        self._yield_sign  = random.choice([-1, 1])
        self._yield_left  = YIELD_FRAMES

    # ─────────────────────────────────────────────────────────────────────────
    # Main control loop
    # ─────────────────────────────────────────────────────────────────────────

    def _control_loop(self):
        self._tick += 1

        # ── Power timer ──────────────────────────────────────────────────────
        if self._powered:
            self._power_timer -= 1
            if self._power_timer <= 0:
                self._powered = False

        # ── Death recovery ───────────────────────────────────────────────────
        if self._dead:
            self._dead_timer -= 1
            if self._dead_timer <= 0:
                self._dead       = False
                self._row, self._col = self._start
                self._powered    = False
                # Teleport back to spawn
                self._teleport_self(*grid_to_world(self._row, self._col))
            self._cmd_pub.publish(Twist())
            return

        # ── Choose next target cell (using pacman.py Player logic) ───────────
        cx, cy = cell_center_world(self._target_row, self._target_col)
        dist_to_target = math.hypot(self._x - cx, self._y - cy)
        
        if dist_to_target < 0.15 and self._tick % 5 == 0:
            class DummyGhost:
                def __init__(self, r, c):
                    self.row = r
                    self.col = c
                    self.dead = False
                    
            dummy_ghosts = {}
            for gid, pos in self._ghost_pos.items():
                if pos:
                    age = self._tick - self._ghost_last_seen.get(gid, 0)
                    if age <= GHOST_TIMEOUT:
                        dummy_ghosts[gid] = DummyGhost(pos[0], pos[1])
                        
            # Sync player state
            self._player.row, self._player.col = self._row, self._col
            
            old_grid = np.copy(self._grid)
            
            # This handles navigation, score tracking, and grid updates!
            self._player.update(dummy_ghosts)
            
            self._target_row = self._player.row
            self._target_col = self._player.col
            self._powered = self._player.powered
            self._score = self._player.score
            
            # Check for pellets that were consumed in this step and teleport them
            for r in range(self._rows):
                for c in range(self._cols):
                    if old_grid[r, c] in (PELLET, POWER) and self._grid[r, c] == EMPTY:
                        self._teleport_pellet(r, c, old_grid[r, c])

        # ── Sideways-yield collision check ───────────────────────────────────
        if self._yield_left <= 0 and self._check_ghost_same_cell():
            self._start_yield()

        # ── Compute cmd_vel ──────────────────────────────────────────────────
        cmd = self._compute_cmd()
        self._cmd_pub.publish(cmd)

        # ── Broadcast location + power state ────────────────────────────────
        loc_msg = Point()
        loc_msg.x = float(self._row)
        loc_msg.y = float(self._col)
        loc_msg.z = 0.0
        self._loc_pub.publish(loc_msg)

        pow_msg = Bool()
        pow_msg.data = self._powered
        self._power_pub.publish(pow_msg)

        # ── NRF24 heartbeat (own location) ───────────────────────────────────
        if self._tick % HEARTBEAT_EVERY == 0:
            pkt = {
                'id':    ['pacman', int(self._tick), int(self._nrf_seq)],
                'diffs': [('heartbeat', -1, int(self._row), int(self._col), int(self._tick))],
                'hop':   0,
            }
            self._nrf_seq += 1
            self._nrf_pub.publish(String(data=json.dumps(pkt)))

        # ── Logging ──────────────────────────────────────────────────────────
        if self._tick % 50 == 0:
            self.get_logger().info(
                f'score={self._score} powered={self._powered} '
                f'cell=({self._row},{self._col}) '
                f'pellets={int(np.sum(np.isin(self._grid, [PELLET, POWER])))}'
            )

    def _compute_cmd(self) -> Twist:
        """
        Omni-directional proportional controller.
        Strafes directly towards target cell.
        """
        cmd = Twist()
        tx, ty = cell_center_world(self._target_row, self._target_col)

        err_x = tx - self._x
        err_y = ty - self._y

        # During yield: add perpendicular offset
        yield_ox = yield_oy = 0.0
        if self._yield_left > 0:
            self._yield_left -= 1
            if self._yield_axis == 'x':
                yield_ox = self._yield_sign * YIELD_OFFSET
            else:
                yield_oy = self._yield_sign * YIELD_OFFSET
                
        err_x += yield_ox
        err_y += yield_oy

        dist = math.hypot(err_x, err_y)

        if dist < CELL_SNAP_DIST:
            return cmd

        # Omni-directional strafing
        cmd.linear.x = (err_x / dist) * BOT_SPEED
        cmd.linear.y = (err_y / dist) * BOT_SPEED
        cmd.angular.z = 0.0

        return cmd

    def _teleport_self(self, x: float, y: float, z: float = SPAWN_Z):
        if not self._set_state.service_is_ready():
            return
        req = SetEntityState.Request()
        state = EntityState()
        state.name = PACMAN_NAME
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.w = 1.0
        req.state = state
        self._set_state.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = PacmanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node._cmd_pub.publish(Twist())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
