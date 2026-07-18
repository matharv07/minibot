#!/usr/bin/env python3
"""
score_visualizer.py — Live terminal dashboard for the Pac-Man Gazebo simulation.

Subscribes to /pacman_bot/stats (std_msgs/String, JSON published by pacman_node)
and renders a rich, colour-coded dashboard in the terminal using ANSI escape codes.

Layout
──────
  ┌──────────────────────────────────────────────────────┐
  │               🕹  PAC-MAN  LIVE  STATS               │
  ├────────────────────┬─────────────────────────────────┤
  │  SCORE             │  POWER STATE                    │
  │  CELL              │  PELLETS LEFT                   │
  │  PELLETS EATEN     │  POWER EATEN  │  GHOSTS EATEN   │
  │  BOT SPEED         │  TICK                           │
  └────────────────────┴─────────────────────────────────┘

Runs as a standalone ROS 2 node.  Launch via pacman.launch.py or directly:
    python3 score_visualizer.py
"""

import json
import os
import sys
import math
import time
import shutil

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ── ANSI colour helpers ───────────────────────────────────────────────────────
RESET  = '\033[0m'
BOLD   = '\033[1m'
DIM    = '\033[2m'
BLINK  = '\033[5m'

# Foreground colours
FG_BLACK   = '\033[30m'
FG_RED     = '\033[91m'
FG_GREEN   = '\033[92m'
FG_YELLOW  = '\033[93m'
FG_BLUE    = '\033[94m'
FG_MAGENTA = '\033[95m'
FG_CYAN    = '\033[96m'
FG_WHITE   = '\033[97m'
FG_ORANGE  = '\033[38;5;214m'

# Background colours
BG_BLACK   = '\033[40m'
BG_BLUE    = '\033[44m'
BG_DARK    = '\033[48;5;17m'     # deep navy
BG_PANEL   = '\033[48;5;234m'   # near-black panel

CLS        = '\033[2J\033[H'    # clear screen + move cursor home


def _col(w: int) -> int:
    """Safe terminal column width."""
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def _fmt_bar(value: int, max_value: int, width: int = 20,
             fill: str = '█', empty: str = '░',
             color: str = FG_GREEN) -> str:
    frac = max(0.0, min(1.0, value / max_value)) if max_value > 0 else 0.0
    filled = int(round(frac * width))
    bar = color + fill * filled + DIM + empty * (width - filled) + RESET
    return f'[{bar}]'


def _power_bar(timer: int, max_timer: int = 40, width: int = 18) -> str:
    if timer <= 0:
        return DIM + '[' + '░' * width + ']' + RESET
    color = FG_CYAN if timer > max_timer // 3 else FG_YELLOW
    return _fmt_bar(timer, max_timer, width, color=color)


class ScoreVisualizer(Node):
    def __init__(self):
        super().__init__('score_visualizer')
        self._stats = {}
        self._last_update = time.time()
        self._total_pellets_start = None  # set on first message
        self._start_time = time.time()
        self._high_score = 0

        self.create_subscription(
            String, '/pacman_bot/stats',
            self._stats_cb, 10
        )
        # Render at 5 Hz even if no new data
        self.create_timer(0.2, self._render)

        # Hide cursor
        sys.stdout.write('\033[?25l')
        sys.stdout.flush()
        self.get_logger().info('score_visualizer started — watching /pacman_bot/stats')

    def _stats_cb(self, msg: String):
        try:
            self._stats = json.loads(msg.data)
            self._last_update = time.time()
            score = self._stats.get('score', 0)
            if score > self._high_score:
                self._high_score = score
            # Set initial pellet total on first message
            if self._total_pellets_start is None:
                eaten = (self._stats.get('pellets_eaten', 0)
                         + self._stats.get('power_eaten', 0))
                self._total_pellets_start = (self._stats.get('pellets_left', 0) + eaten)
        except Exception:
            pass

    def _render(self):
        s = self._stats
        if not s:
            # Waiting for data
            sys.stdout.write(CLS)
            sys.stdout.write(
                f'\n{BOLD}{FG_YELLOW}  🟡 Waiting for pacman_node to start...'
                f'{RESET}\n'
            )
            sys.stdout.flush()
            return

        cols = _col(0)
        box_w = min(cols - 2, 64)
        inner = box_w - 2

        score        = s.get('score', 0)
        powered      = s.get('powered', False)
        power_timer  = s.get('power_timer', 0)
        row          = s.get('row', 0)
        col_g        = s.get('col', 0)
        pellets_left = s.get('pellets_left', 0)
        pellets_eat  = s.get('pellets_eaten', 0)
        power_eat    = s.get('power_eaten', 0)
        ghosts_eat   = s.get('ghosts_eaten', 0)
        speed        = s.get('speed', 0.0)
        tick         = s.get('tick', 0)
        target       = s.get('target', [row, col_g])

        total = self._total_pellets_start or max(1, pellets_left + pellets_eat + power_eat)
        elapsed = time.time() - self._start_time
        age = time.time() - self._last_update

        # ── Build display ─────────────────────────────────────────────────────
        lines = []

        # Header
        title = '🕹  PAC-MAN  LIVE  STATS'
        pad = (inner - len(title)) // 2
        lines.append(f'{BG_DARK}{BOLD}{FG_YELLOW}' + '─' * box_w + RESET)
        lines.append(f'{BG_DARK}{BOLD}{FG_YELLOW}│' + ' ' * pad + title
                     + ' ' * (inner - pad - len(title)) + '│' + RESET)
        lines.append(f'{BG_DARK}{BOLD}{FG_YELLOW}' + '─' * box_w + RESET)

        def row_line(label: str, value: str, color: str = FG_WHITE) -> str:
            lbl = f'{DIM}{FG_CYAN}{label:<20}{RESET}'
            val = f'{BOLD}{color}{value}{RESET}'
            content = lbl + val
            # Strip ANSI for length calculation
            import re
            ansi_escape = re.compile(r'\033\[[0-9;]*m')
            raw_len = len(ansi_escape.sub('', content))
            pad_r = max(0, inner - raw_len)
            return f'{BG_PANEL}│ {content}{" " * pad_r}│{RESET}'

        # Score
        score_color = FG_ORANGE if score > 0 else FG_WHITE
        last_points = s.get('last_points', 0)
        score_text = f'{score:>8,}   (best: {self._high_score:,})'
        if last_points > 0:
            score_text += f'   [ +{last_points} pts ]'
        lines.append(row_line('SCORE', score_text, score_color))

        # Power state
        if powered:
            pow_str = (f'{BOLD}{BLINK}{FG_CYAN}⚡ POWERED{RESET}  '
                       + _power_bar(power_timer)
                       + f'  {power_timer} ticks')
        else:
            pow_str = f'{DIM}inactive{RESET}'
        lines.append(row_line('POWER STATE', pow_str, FG_CYAN if powered else FG_WHITE))

        # Position
        lines.append(row_line('CELL',
                               f'({row:>2}, {col_g:>2})  →  ({target[0]:>2}, {target[1]:>2})',
                               FG_GREEN))

        # Pellets progress bar
        consumed = total - pellets_left
        bar = _fmt_bar(consumed, total, width=22, color=FG_YELLOW)
        lines.append(row_line('PELLETS LEFT',
                               f'{pellets_left:>4} / {total:<4} {bar}',
                               FG_YELLOW))

        lines.append(row_line('PELLETS EATEN',  f'{pellets_eat:>4}', FG_WHITE))
        lines.append(row_line('POWER EATEN',    f'{power_eat:>4}', FG_MAGENTA))
        lines.append(row_line('GHOSTS EATEN',   f'{ghosts_eat:>4}', FG_RED))

        lines.append(row_line('BOT SPEED',
                               f'{speed:.3f} m/s', FG_GREEN))

        # Timing
        elapsed_str = f'{int(elapsed // 60):02d}:{int(elapsed % 60):02d}'
        lines.append(row_line('ELAPSED',
                               f'{elapsed_str}   tick {tick}', FG_WHITE))

        # Data freshness
        fresh = age < 1.5
        age_col = FG_GREEN if fresh else FG_RED
        age_str = f'{age:.1f}s ago'
        lines.append(row_line('DATA AGE', age_str, age_col))

        # Footer
        lines.append(f'{BG_DARK}{FG_YELLOW}' + '─' * box_w + RESET)
        lines.append(f'{DIM}  Ctrl+C to exit  │  ROS 2 topic: /pacman_bot/stats{RESET}')

        # Write all at once
        out = CLS + '\n'.join(lines) + '\n'
        sys.stdout.write(out)
        sys.stdout.flush()

    def destroy_node(self):
        # Restore cursor
        sys.stdout.write('\033[?25h\n')
        sys.stdout.flush()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ScoreVisualizer()
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
