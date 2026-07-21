#!/usr/bin/env python3
"""
maze_generator.py — Shared arena/maze parameters for the Pac-Man Gazebo simulation.

Grid coordinate system
  row increases → +Y in world  (row 0 = top of arena, row ROWS-1 = bottom)
  col increases → +X in world  (col 0 = left,          col COLS-1 = right)

Cell pitch = CELL_SIZE metres.  Bots travel along cell centre-lines.

NRF24L01 comm radius: RADIUS_M = 4.8 m  (from ghost.py: RADIUS=12 cells @ 0.35m/cell
  => 4.2 m; we round up to 4.8 m as per user spec)
"""

import random
import math
import numpy as np
from typing import Tuple, List

# ── Arena dimensions ──────────────────────────────────────────────────────────
ROWS        = 33          # maze rows   (matches pacman.py stage-4: 33 rows)
COLS        = 41          # maze cols   (matches pacman.py stage-4: 41 cols)
CELL_SIZE   = 0.35        # metres per grid cell (matches prev sessions)
N_GHOSTS    = 7           # 7 ghosts — one per GHOST_COLORS_RGB entry
N_POWER     = 28          # number of power pellets

# ── Cell types ────────────────────────────────────────────────────────────────
WALL   = 1
EMPTY  = 0
PELLET = 2
POWER  = 3

# ── Bot physical parameters (from URDF) ──────────────────────────────────────
BOT_RADIUS   = 0.045      # m  (cylinder collision radius from minibot_game.xacro)
BOT_HEIGHT   = 0.07       # m  (approx. chassis height)
SPAWN_Z      = 0.035      # m  (nominal spawn height from launch file)

# Float pellets higher above the bots (elevated to not interfere with sonar/bots)
PELLET_Z     = SPAWN_Z + 0.100   # elevated to avoid sonar/bots
POWER_Z      = SPAWN_Z + 0.110   # slightly higher than regular pellets

# ── NRF24L01 communication radius ────────────────────────────────────────────
NRF_RADIUS_CELLS  = 12               # from ghost.py RADIUS constant
NRF_RADIUS_M      = NRF_RADIUS_CELLS * CELL_SIZE   # ≈ 4.2 m (spec says ~4.8)

# ── Gazebo model name prefixes ────────────────────────────────────────────────
PACMAN_NAME   = "pacman_bot"
GHOST_NAMES   = [f"ghost_{i}" for i in range(N_GHOSTS)]

# ── Ghost colors (R,G,B) matching pacman.py GHOST_COLORS ─────────────────────
GHOST_COLORS_RGB = [
    (0.86, 0.12, 0.12),   # RED
    (1.00, 0.39, 0.71),   # PINK
    (0.00, 0.86, 0.86),   # CYAN
    (1.00, 0.63, 0.12),   # ORANGE
    (0.70, 0.00, 0.70),   # PURPLE
    (0.00, 0.71, 0.31),   # GREEN
    (0.86, 0.86, 0.00),   # YELLOW
]

# ── Pacman color: bright yellow ───────────────────────────────────────────────
PACMAN_COLOR_RGB   = (1.00, 0.86, 0.00)
PACMAN_COLOR_POWERED = (0.00, 0.47, 1.00)  # blue when powered (visual override)

# ── Directions ────────────────────────────────────────────────────────────────
UP    = (-1,  0)
DOWN  = ( 1,  0)
LEFT  = ( 0, -1)
RIGHT = ( 0,  1)
DIRS  = [UP, DOWN, LEFT, RIGHT]


# ─────────────────────────────────────────────────────────────────────────────
# World ↔ Grid coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def grid_to_world(row: int, col: int, z: float = SPAWN_Z) -> Tuple[float, float, float]:
    """Return (x, y, z) world coordinates for the centre of grid cell (row, col)."""
    # Arena origin at (0, 0); row=0 col=0 maps to x = -(COLS/2)*CELL, y = +(ROWS/2)*CELL
    x = (col - COLS / 2.0 + 0.5) * CELL_SIZE
    y = (ROWS / 2.0 - row - 0.5) * CELL_SIZE
    return (x, y, z)


def world_to_grid(x: float, y: float) -> Tuple[int, int]:
    """Convert world (x, y) to nearest grid (row, col)."""
    col = int(round(x / CELL_SIZE + COLS / 2.0 - 0.5))
    row = int(round(ROWS / 2.0 - y / CELL_SIZE - 0.5))
    row = max(0, min(ROWS - 1, row))
    col = max(0, min(COLS - 1, col))
    return (row, col)


def cell_center_world(row: int, col: int) -> Tuple[float, float]:
    """Return (x, y) world coordinates of cell centre (no z)."""
    x, y, _ = grid_to_world(row, col, 0.0)
    return (x, y)


import sys
import os

def _find_pacmanbot_dir():
    try:
        # Try ROS 2 install directory first (when running via ros2 launch)
        from ament_index_python.packages import get_package_share_directory
        pkg_share = get_package_share_directory('minibot')
        d = os.path.join(pkg_share, 'pacmanbot')
        if os.path.isdir(d):
            return d
    except Exception:
        pass
        
    # Fallback for direct script execution in the source tree
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pacmanbot'))

_PACMAN_DIR = _find_pacmanbot_dir()
if _PACMAN_DIR not in sys.path:
    sys.path.insert(0, _PACMAN_DIR)

from pacman import generate_map as pacman_generate_map
from pacman import Game

def generate_map(rows: int = ROWS, cols: int = COLS, n_power: int = N_POWER,
                 random_spawn: bool = False, seed: int | None = None) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Delegate map generation directly to pacman.py to ensure 100% parity.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    return pacman_generate_map(rows, cols, n_power, random_spawn)


def compute_ghost_starts(grid: np.ndarray, player_start: Tuple[int, int], n: int) -> List[Tuple[int, int]]:
    """
    Delegate ghost placement directly to the logic used in pacman.py.
    """
    open_cells = np.argwhere(grid != WALL)
    pac_pos    = np.array(player_start)
    dist_pac   = np.sum(np.abs(open_cells - pac_pos), axis=1)
    min_dist   = np.full(len(open_cells), np.inf)
    available  = np.ones(len(open_cells), dtype=bool)

    first_idx = int(np.argmax(dist_pac))
    starts    = [tuple(open_cells[first_idx])]
    available[first_idx] = False

    for _ in range(n - 1):
        last = np.array(starts[-1])
        d    = np.sum(np.abs(open_cells - last), axis=1)
        min_dist = np.minimum(min_dist, d)
        scores   = np.minimum(dist_pac, min_dist)
        scores[~available] = -1
        best = int(np.argmax(scores))
        starts.append(tuple(open_cells[best]))
        available[best] = False

    return starts[:n]


def pellet_positions(grid: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return list of (row, col, cell_type) for all pellet/power cells."""
    result = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] in (PELLET, POWER):
                result.append((r, c, int(grid[r, c])))
    return result


def wall_positions(grid: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (row, col) for all wall cells."""
    return [tuple(x) for x in np.argwhere(grid == WALL)]
