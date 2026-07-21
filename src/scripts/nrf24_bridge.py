#!/usr/bin/env python3
"""
nrf24_bridge.py — Simulated NRF24L01 transceiver bridge for Pac-Man Gazebo.

Architecture
────────────
Each bot publishes its outgoing radio packet on:
    /nrf24/<sender_name>/tx          (std_msgs/String, JSON payload)

This bridge node subscribes to ALL tx topics and forwards packets to:
    /nrf24/<recipient_name>/rx       (std_msgs/String, JSON payload)

…but ONLY when the sender and recipient are within NRF_RADIUS_M of each other
(enforced by comparing their latest /odom poses).

Topic map
─────────
  /nrf24/pacman_bot/tx    → bridge → /nrf24/ghost_*/rx
  /nrf24/ghost_*/tx       → bridge → /nrf24/pacman_bot/rx AND /nrf24/ghost_*/rx
                                      (ghost-to-ghost limited to NRF_RADIUS_M)

Special broadcasts (ghost relay)
  Ghosts broadcast their full NRF24 diffs on /nrf24/<ghost>/tx using the same
  JSON schema as ghost.py _broadcast() — but limited to NRF_RADIUS_M range.

Parameters (ROS params)
───────────────────────
  nrf_radius_m   : float  — comm radius in metres (default: 4.8)
  bot_names      : list   — all bot entity names to bridge (default: pacman_bot + ghost_0..3)
"""

import json
import math
import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry

from maze_generator import (
    NRF_RADIUS_M, PACMAN_NAME, GHOST_NAMES, N_GHOSTS
)


class NRF24Bridge(Node):

    def __init__(self):
        super().__init__('nrf24_bridge')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('nrf_radius_m', NRF_RADIUS_M)
        self.declare_parameter('bot_names', [PACMAN_NAME] + GHOST_NAMES)

        self._radius = self.get_parameter('nrf_radius_m').value
        self._bot_names: list[str] = list(self.get_parameter('bot_names').value)

        # ── State ────────────────────────────────────────────────────────────
        # Latest (x, y) world position for each bot
        self._positions: dict[str, tuple[float, float]] = {}

        # ── Subscribers: one odom + one tx per bot ───────────────────────────
        self._tx_subs: dict[str, object] = {}
        self._rx_pubs: dict[str, object] = {}

        for name in self._bot_names:
            # odom → track position
            self.create_subscription(
                Odometry,
                f'/{name}/odom',
                self._make_odom_cb(name),
                10
            )
            # tx  → inbound radio packet
            self.create_subscription(
                String,
                f'/nrf24/{name}/tx',
                self._make_tx_cb(name),
                20
            )
            # rx  → outbound radio packet to this bot
            self._rx_pubs[name] = self.create_publisher(
                String,
                f'/nrf24/{name}/rx',
                20
            )

        self.get_logger().info(
            f'NRF24 bridge ready — radius={self._radius:.2f} m, '
            f'bots={self._bot_names}'
        )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _make_odom_cb(self, name: str):
        def cb(msg: Odometry):
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self._positions[name] = (x, y)
        return cb

    def _make_tx_cb(self, sender: str):
        def cb(msg: String):
            self._route(sender, msg.data)
        return cb

    def _route(self, sender: str, raw: str):
        """Forward raw packet from sender to all in-range recipients."""
        src_pos = self._positions.get(sender)
        if src_pos is None:
            # position not yet known — broadcast to everyone (startup grace)
            for name in self._bot_names:
                if name != sender:
                    self._rx_pubs[name].publish(String(data=raw))
            return

        sent = 0
        for name in self._bot_names:
            if name == sender:
                continue
            dst_pos = self._positions.get(name)
            if dst_pos is None:
                # destination not yet localised — deliver anyway (startup)
                self._rx_pubs[name].publish(String(data=raw))
                sent += 1
                continue
            dist = math.hypot(src_pos[0] - dst_pos[0], src_pos[1] - dst_pos[1])
            if dist <= self._radius or name == PACMAN_NAME or sender == PACMAN_NAME:
                self._rx_pubs[name].publish(String(data=raw))
                sent += 1




def main(args=None):
    rclpy.init(args=args)
    node = NRF24Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
