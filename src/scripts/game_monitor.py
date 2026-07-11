#!/usr/bin/env python3
"""
game_monitor.py — prints Pac-Man game metrics to the terminal every second.

Subscribes to:
  /game/score              std_msgs/Int32
  /game/state              std_msgs/String   ('playing' | 'win')
  /game/powered            std_msgs/Bool
  /game/steps              std_msgs/Int32
  /game/pellets_remaining  std_msgs/Int32
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool, String, Int32MultiArray

GHOST_COUNT = 7   # all ghosts alive until death mechanic is added

RESET   = '\033[0m'
BOLD    = '\033[1m'
YELLOW  = '\033[93m'
CYAN    = '\033[96m'
GREEN   = '\033[92m'
RED     = '\033[91m'
MAGENTA = '\033[95m'
GREY    = '\033[90m'
CLEAR   = '\033[2J\033[H'


class GameMonitorNode(Node):
    def __init__(self):
        super().__init__('game_monitor')

        self.score     = 0
        self.state     = 'waiting...'
        self.powered   = False
        self.steps     = 0
        self.pellet_status = None   # Will store [consumed, total, p_consumed, p_total]

        self.create_subscription(Int32,  '/game/score',             self._score,   10)
        self.create_subscription(String, '/game/state',             self._state_cb,10)
        self.create_subscription(Bool,   '/game/powered',           self._power,   10)
        self.create_subscription(Int32,  '/game/steps',             self._steps,   10)
        self.create_subscription(Int32MultiArray, '/game/pellets_remaining', self._pellets, 10)

        self.create_timer(1.0, self._print)
        self.get_logger().info('game_monitor running — waiting for game data…')

    def _score(self,   m): self.score   = m.data
    def _state_cb(self,m): self.state   = m.data
    def _power(self,   m): self.powered = m.data
    def _steps(self,   m): self.steps   = m.data
    def _pellets(self, m): self.pellet_status = m.data

    def _print(self):
        power_str = f'{MAGENTA}⚡ POWERED{RESET}' if self.powered else f'{GREY}normal{RESET}'
        state_col = GREEN if self.state == 'win' else (YELLOW if self.state == 'playing' else GREY)
        if self.pellet_status:
            pellet_str = f"{self.pellet_status[0]}/{self.pellet_status[1]}"
            power_p_str = f"{self.pellet_status[2]}/{self.pellet_status[3]}"
        else:
            pellet_str = "…"
            power_p_str = "…"

        print(CLEAR, end='')
        print(f'{BOLD}{CYAN}╔══════════════════════════════╗{RESET}')
        print(f'{BOLD}{CYAN}║   PAC-MAN  GAME  MONITOR     ║{RESET}')
        print(f'{BOLD}{CYAN}╚══════════════════════════════╝{RESET}')
        print(f'  {BOLD}Score   {RESET}: {YELLOW}{self.score:>6}{RESET}')
        print(f'  {BOLD}Steps   {RESET}: {self.steps:>6}')
        print(f'  {BOLD}Pellets {RESET}: {RED}{pellet_str:>9}{RESET} consumed')
        print(f'  {BOLD}Powers  {RESET}: {MAGENTA}{power_p_str:>9}{RESET} consumed')
        print(f'  {BOLD}Ghosts  {RESET}: {RED}{GHOST_COUNT}/7{RESET}  alive')
        print(f'  {BOLD}State   {RESET}: {state_col}{self.state}{RESET}')
        print(f'  {BOLD}Aura    {RESET}: {power_str}')
        print()


def main(args=None):
    rclpy.init(args=args)
    node = GameMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
