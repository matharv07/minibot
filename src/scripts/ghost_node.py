#!/usr/bin/env python3
"""
ghost_node.py — Minimal ghost bot ROS 2 node for Pac-Man Gazebo simulation.

Each ghost_node instance:
  • Subscribes to its own /nrf24/<name>/rx for incoming radio packets
  • Publishes to /nrf24/<name>/tx for outgoing radio packets (location relay)
  • Subscribes to /pacman_bot/location and /pacman_bot/power_state (topics)
  • Publishes odom-derived ghost location + ghost-to-ghost NRF24 relay
  • Moves along grid centre-lines using the planar_move plugin

Ghost AI is NOT implemented yet — ghosts are spawned and hold position until
the full ghost_node AI is added in a future iteration.

Topics consumed
───────────────
  /<name>/odom                 (nav_msgs/Odometry)        — own position
  /nrf24/<name>/rx             (std_msgs/String, JSON)    — inbound NRF24
  /pacman_bot/location         (geometry_msgs/Point)      — pacman grid pos
  /pacman_bot/power_state      (std_msgs/Bool)            — pacman powered?

Topics published
────────────────
  /<name>/cmd_vel              (geometry_msgs/Twist)      — motion command
  /nrf24/<name>/tx             (std_msgs/String, JSON)    — outbound NRF24
"""
import json
import sys
import os

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String, Bool

# Ensure maze_generator is importable from the scripts directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from maze_generator import (
    PACMAN_NAME, GHOST_NAMES, NRF_RADIUS_M,
    world_to_grid, CELL_SIZE, SPAWN_Z
)


class GhostNode(Node):

    def __init__(self, name: str, ghost_id: int):
        super().__init__(f'ghost_node_{name}')
        self._name      = name
        self._ghost_id  = ghost_id

        # ── State ────────────────────────────────────────────────────────────
        self._x = 0.0
        self._y = 0.0
        self._row = 0
        self._col = 0
        self._pacman_pos: tuple[int, int] | None = None
        self._pacman_powered: bool = False
        self._frame: int = 0

        # NRF24 message dedup
        self._seen_msg_ids: set = set()

        # ── Publishers ───────────────────────────────────────────────────────
        self._cmd_pub  = self.create_publisher(Twist, f'/{name}/cmd_vel', 10)
        self._nrf_pub  = self.create_publisher(String, f'/nrf24/{name}/tx', 20)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Odometry, f'/{name}/odom',  self._odom_cb,  10)
        self.create_subscription(String,   f'/nrf24/{name}/rx', self._nrf_rx_cb, 20)
        self.create_subscription(Point,    '/pacman_bot/location',    self._pac_loc_cb, 10)
        self.create_subscription(Bool,     '/pacman_bot/power_state', self._pac_pow_cb, 10)

        # ── Control loop ─────────────────────────────────────────────────────
        self._timer = self.create_timer(1.0 / 30.0, self._control_loop)  # 30 Hz

        self.get_logger().info(f'Ghost node "{name}" (id={ghost_id}) ready')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._row, self._col = world_to_grid(self._x, self._y)

    def _nrf_rx_cb(self, msg: String):
        """Handle incoming NRF24 packet from bridge."""
        try:
            pkt = json.loads(msg.data)
        except Exception:
            return
        mid = pkt.get('id')
        if mid is not None:
            key = str(mid)
            if key in self._seen_msg_ids:
                return
            self._seen_msg_ids.add(key)
            # Rolling prune
            if len(self._seen_msg_ids) > 500:
                to_del = list(self._seen_msg_ids)[:250]
                for k in to_del:
                    self._seen_msg_ids.discard(k)

        # Ghost-node just logs received packets for now (AI TBD)
        # Future: process pacman sighting, ghost positions, etc.

    def _pac_loc_cb(self, msg: Point):
        """Receive pacman location broadcast (topic, not NRF24)."""
        self._pacman_pos = (int(round(msg.x)), int(round(msg.y)))

    def _pac_pow_cb(self, msg: Bool):
        """Receive pacman power-state broadcast (topic)."""
        self._pacman_powered = msg.data

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_loop(self):
        self._frame += 1

        # ── Ghost AI not yet implemented — hold spawn position ────────────────
        # Publish zero velocity so the planar_move plugin doesn't drift.
        self._cmd_pub.publish(Twist())

        # ── NRF24 heartbeat (own position) ───────────────────────────────────
        if self._frame % 5 == 0:
            pkt = {
                'id':    [int(self._ghost_id), int(self._frame), 0],
                'diffs': [
                    ('heartbeat', int(self._ghost_id),
                     int(self._row), int(self._col), int(self._frame))
                ],
                'hop': 0,
            }
            self._nrf_pub.publish(String(data=json.dumps(pkt)))

    def _stop(self):
        if rclpy.ok():
            self._cmd_pub.publish(Twist())



def main(args=None):
    rclpy.init(args=args)

    # Read ghost_id and ghost_name from --ros-args -p params passed by launch
    import sys as _sys
    gid   = 0
    gname = 'ghost_0'
    raw_args = _sys.argv[1:]
    for i, a in enumerate(raw_args):
        if a in ('-p', '--param') and i + 1 < len(raw_args):
            kv = raw_args[i + 1]
            if kv.startswith('ghost_id:='):
                gid = int(kv.split(':=', 1)[1])
            elif kv.startswith('ghost_name:='):
                gname = kv.split(':=', 1)[1]

    ghost = GhostNode(gname, gid)
    try:
        rclpy.spin(ghost)
    except KeyboardInterrupt:
        ghost._stop()
    finally:
        ghost.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
