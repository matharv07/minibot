#!/usr/bin/env python3
"""
generate_maze.py — Converts a pacmanbot maze grid into a Gazebo SDF world.

Called at launch-time by pacman.launch.py.  Writes an SDF file to /tmp and prints
a JSON blob to stdout with:
  - "world_path":  path to the generated .world file
  - "spawns":      dict  bot_name → {"x": ..., "y": ...}

Optimised for fast Gazebo loading:
  - Adjacent wall cells are merged into horizontal strips (reduces ~700 → ~100 models)
  - No pellet models (game logic, not physics objects)
  - Lane markers only at bot spawn positions (8 markers instead of ~5000)
"""
import sys
import os
import json
import random
import tempfile
import numpy as np

# ── Import generate_map from pacmanbot ──────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_PATHS = [
    os.path.join(_SCRIPT_DIR, '..', 'pacmanbot'),
    os.path.join(_SCRIPT_DIR, '..', 'share', 'minibot', 'pacmanbot'),
]
for _p in _CANDIDATE_PATHS:
    _p = os.path.normpath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Mock pygame and heavy deps so we can import generate_map headlessly
import types
_fake_pygame = types.ModuleType('pygame')
_fake_pygame.Rect = lambda *a, **kw: None
_fake_pygame.init = lambda: None
_fake_pygame.font = types.ModuleType('pygame.font')
_fake_pygame.font.init = lambda: None
_fake_pygame.font.SysFont = lambda *a, **kw: None
_fake_pygame.font.Font = lambda *a, **kw: None
_fake_pygame.display = types.ModuleType('pygame.display')
_fake_pygame.display.set_mode = lambda *a, **kw: None
_fake_pygame.display.set_caption = lambda *a, **kw: None
_fake_pygame.draw = types.ModuleType('pygame.draw')
_fake_pygame.Surface = lambda *a, **kw: None
_fake_pygame.SRCALPHA = 0
_fake_pygame.QUIT = 0; _fake_pygame.KEYDOWN = 0; _fake_pygame.MOUSEBUTTONDOWN = 0
_fake_pygame.K_w = 0; _fake_pygame.K_a = 0; _fake_pygame.K_s = 0; _fake_pygame.K_d = 0
_fake_pygame.K_UP = 0; _fake_pygame.K_DOWN = 0; _fake_pygame.K_LEFT = 0; _fake_pygame.K_RIGHT = 0
_fake_pygame.K_r = 0
_fake_pygame.K_0 = 0; _fake_pygame.K_1 = 0; _fake_pygame.K_2 = 0; _fake_pygame.K_3 = 0
_fake_pygame.K_4 = 0; _fake_pygame.K_5 = 0; _fake_pygame.K_6 = 0
sys.modules['pygame'] = _fake_pygame
sys.modules['pygame.font'] = _fake_pygame.font
sys.modules['pygame.display'] = _fake_pygame.display
sys.modules['pygame.draw'] = _fake_pygame.draw

for _mod_name in ('torch', 'ghost', 'pathfinder', 'cbba', 'beliefmap',
                  'allocator', 'obs', 'net', 'curriculum', 'setup_dependencies'):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _mod_name == 'ghost':
            _stub.Ghost = type('Ghost', (), {'__init__': lambda *a, **kw: None})
            _stub.UNKNOWN = -1
        if _mod_name == 'pathfinder':
            _stub.build_scipy_graph = lambda *a, **kw: None
            _stub._SCIPY_AVAILABLE = False
            _stub.get_scipy_graph = lambda *a, **kw: None
        if _mod_name == 'curriculum':
            from dataclasses import dataclass
            @dataclass(frozen=True)
            class _Stage:
                rows: int; cols: int; n_ghosts: int; n_power: int
                advance_return: float; min_updates: int
            _stub.STAGES = [
                _Stage(7,9,2,2,42.0,150), _Stage(13,17,3,6,40.0,350),
                _Stage(21,27,5,14,28.0,360), _Stage(27,33,6,24,22.0,400),
                _Stage(33,41,7,28,float('inf'),0),
            ]
        if _mod_name == 'net':
            _stub.GhostActor = type('GhostActor', (), {})
        if _mod_name == 'setup_dependencies':
            _stub.main = lambda: None
        sys.modules[_mod_name] = _stub

from pacman import generate_map, WALL, EMPTY, PELLET, POWER  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────────
ROWS = 33
COLS = 41
N_GHOSTS = 7
N_POWER = 28

BOT_SIZE     = 0.10    # metres — minibot footprint
LANE_WIDTH   = 0.09    # bot + 0.01 m gap (tighter to fit 8 bots in cell)
WALL_THICK   = 0.15    # 1.5 × bot_size
WALL_HEIGHT  = 0.105   # 1.5 × bot height (~70 mm)

PASSAGE_SPAN = 7 * LANE_WIDTH + BOT_SIZE          # 0.73 m
CELL_PITCH   = PASSAGE_SPAN + WALL_THICK           # 0.88 m

# Channel offsets (diagonal — unique X and Y per bot)
# With LANE_WIDTH=0.09, max offset for ghost 6 is 4 * 0.09 = 0.36m
# 0.36m + 0.05m = 0.41m < 0.44m (half pitch) -> fits perfectly!
CHANNEL_OFFSETS = [
    ( 0,  0),   # pacman  — centre
    (+1, +1),   # ghost 0
    (-1, -1),   # ghost 1
    (+2, +2),   # ghost 2
    (-2, -2),   # ghost 3
    (+3, +3),   # ghost 4
    (-3, -3),   # ghost 5
    (+4, +4),   # ghost 6
]

# Bot colours  (R, G, B) 0-255 — from pacman.py
BOT_COLORS = [
    (255, 220,   0),   # pacman — yellow
    (220,  30,  30),   # ghost 0 — red
    (255, 100, 180),   # ghost 1 — pink
    (  0, 220, 220),   # ghost 2 — cyan
    (255, 160,  30),   # ghost 3 — orange
    (180,   0, 180),   # ghost 4 — purple
    (  0, 180,  80),   # ghost 5 — green
    (255, 255, 255),   # ghost 6 — white
]

BOT_NAMES = ['pacman'] + [f'ghost_{i}' for i in range(N_GHOSTS)]

# ── Coordinate helpers ──────────────────────────────────────────────────
def cell_centre(row: int, col: int) -> tuple:
    x = col * CELL_PITCH + CELL_PITCH / 2.0
    y = -(row * CELL_PITCH + CELL_PITCH / 2.0)
    return (x, y)

def channel_pos(row: int, col: int, channel_idx: int) -> tuple:
    cx, cy = cell_centre(row, col)
    dx, dy = CHANNEL_OFFSETS[channel_idx]
    return (cx + dx * LANE_WIDTH, cy + dy * LANE_WIDTH)

# ── Wall merging — horizontal strip scan ────────────────────────────────
def merge_walls_horizontal(grid):
    rows, cols = grid.shape
    strips = []
    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r, c] == WALL:
                c_start = c
                while c < cols and grid[r, c] == WALL:
                    c += 1
                strips.append((r, c_start, c - 1))
            else:
                c += 1
    return strips

# ── SDF generation ─────────────────────────────────────────────────────
PELLET_Z   = 0.085   # float height — above bot body (~0.07 m)
POWER_Z    = 0.092
PELLET_S   = 0.030   # diamond half-side
POWER_R    = 0.026   # sphere radius

def _sdf_box(name, x, y, z, sx, sy, sz, r, g, b):
    return (f'<model name="{name}"><static>true</static>'
            f'<pose>{x} {y} {z} 0 0 0</pose>'
            f'<link name="link">'
            f'<collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>'
            f'<visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
            f'<material><ambient>{r} {g} {b} 1</ambient>'
            f'<diffuse>{r} {g} {b} 1</diffuse></material>'
            f'</visual></link></model>\n')

def _sdf_pellet(name, x, y):
    """White floating diamond (box rotated 45° on Z)."""
    s = PELLET_S
    return (f'<model name="{name}"><static>true</static>'
            f'<pose>{x} {y} {PELLET_Z} 0 0 0.7854</pose>'
            f'<link name="link">'
            f'<visual name="v"><geometry><box><size>{s} {s} {s}</size></box></geometry>'
            f'<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>'
            f'<emissive>0.4 0.4 0.4 1</emissive></material>'
            f'</visual></link></model>\n')

def _sdf_power(name, x, y):
    """Gold glowing sphere for power pellets."""
    r = POWER_R
    return (f'<model name="{name}"><static>true</static>'
            f'<pose>{x} {y} {POWER_Z} 0 0 0</pose>'
            f'<link name="link">'
            f'<visual name="v"><geometry><sphere><radius>{r}</radius></sphere></geometry>'
            f'<material><ambient>1.0 0.84 0.0 1</ambient><diffuse>1.0 0.84 0.0 1</diffuse>'
            f'<emissive>0.6 0.5 0.0 1</emissive></material>'
            f'</visual></link></model>\n')


def generate_world(grid, spawns_dict):
    rows, cols = grid.shape
    parts = []
    idx = 0

    # ── Ground plane ────────────────────────────────────────────────
    gx = cols * CELL_PITCH / 2.0
    gy = -(rows * CELL_PITCH / 2.0)
    ground_size = max(cols, rows) * CELL_PITCH + 4.0
    parts.append(_sdf_box('ground', gx, gy, -0.005, ground_size, ground_size, 0.01, 0.02, 0.02, 0.02))

    # ── Merged wall strips ──────────────────────────────────────────
    strips = merge_walls_horizontal(grid)
    for r, c0, c1 in strips:
        n_cells = c1 - c0 + 1
        x0 = c0 * CELL_PITCH + CELL_PITCH / 2.0
        x1 = c1 * CELL_PITCH + CELL_PITCH / 2.0
        cx = (x0 + x1) / 2.0
        cy = -(r * CELL_PITCH + CELL_PITCH / 2.0)
        sx = n_cells * CELL_PITCH
        parts.append(_sdf_box(f'w{idx}', cx, cy, WALL_HEIGHT / 2.0, sx, CELL_PITCH, WALL_HEIGHT, 0.04, 0.04, 0.24))
        idx += 1

    # ── Pellets (white diamonds) and power pellets (gold spheres) ───
    # Placement is already determined by generate_map() — we just read the grid.
    for r in range(rows):
        for c in range(cols):
            cx, cy = cell_centre(r, c)
            if grid[r, c] == PELLET:
                parts.append(_sdf_pellet(f'pel_{r}_{c}', cx, cy))
            elif grid[r, c] == POWER:
                parts.append(_sdf_power(f'pow_{r}_{c}', cx, cy))

    # ── Lane markers at spawn positions only ────────────────────────
    MARKER_H = 0.003
    MARKER_SIZE = 0.04
    for bot_name, sp in spawns_dict.items():
        ch_idx = BOT_NAMES.index(bot_name)
        cr, cg, cb = BOT_COLORS[ch_idx]
        parts.append(_sdf_box(f'mk_{bot_name}', sp['x'], sp['y'], MARKER_H / 2.0, MARKER_SIZE, MARKER_SIZE, MARKER_H, cr/255.0, cg/255.0, cb/255.0))

    # ── Assemble SDF ────────────────────────────────────────────────
    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="pacman_maze">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <light name="sun" type="directional">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 50 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>0 0 -1</direction>
    </light>
    <scene>
      <ambient>0.4 0.4 0.5 1</ambient>
      <background>0.0 0.0 0.05 1</background>
      <shadows>false</shadows>
    </scene>
    <gravity>0 0 -9.81</gravity>
{"".join(parts)}
  </world>
</sdf>
"""
    return sdf

def compute_spawns(grid, player_start):
    rows, cols = grid.shape
    open_cells = list(map(tuple, np.argwhere(grid != WALL)))
    spawns = {}
    px, py = channel_pos(player_start[0], player_start[1], 0)
    spawns['pacman'] = {'x': round(px, 4), 'y': round(py, 4)}
    #ghost placement - same logic as pacman.py
    pac_pos = np.array(player_start)
    oc_arr = np.array(open_cells)
    dist_pac = np.sum(np.abs(oc_arr - pac_pos), axis=1)
    min_dist_to_ghosts = np.full(len(oc_arr), np.inf)
    available = np.ones(len(oc_arr), dtype=bool)
    first_idx = int(np.argmax(dist_pac))
    ghost_starts = [tuple(oc_arr[first_idx])]
    available[first_idx] = False
    for _ in range(N_GHOSTS - 1):
        last_placed = np.array(ghost_starts[-1])
        dist_to_last = np.sum(np.abs(oc_arr - last_placed), axis=1)
        min_dist_to_ghosts = np.minimum(min_dist_to_ghosts, dist_to_last)
        scores = np.minimum(dist_pac, min_dist_to_ghosts)
        scores[~available] = -1
        best_idx = int(np.argmax(scores))
        ghost_starts.append(tuple(oc_arr[best_idx]))
        available[best_idx] = False
    for i, gs in enumerate(ghost_starts):
        gx, gy = channel_pos(gs[0], gs[1], i + 1)
        spawns[f'ghost_{i}'] = {'x': round(gx, 4), 'y': round(gy, 4)}
    return spawns

def main():
    grid, player_start = generate_map(rows=ROWS, cols=COLS, n_power=N_POWER, random_spawn=False)
    spawns = compute_spawns(grid, player_start)
    sdf = generate_world(grid, spawns)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='_pacman_maze.world', delete=False, prefix='game_')
    tmp.write(sdf)
    tmp.flush()
    tmp.close()
    result = {
        'world_path':   tmp.name,
        'spawns':       spawns,
        'grid':         grid.tolist(),         # full grid for pacman_node
        'player_start': [int(player_start[0]), int(player_start[1])],
        'cell_pitch':   CELL_PITCH,
    }
    print(json.dumps(result))

if __name__ == '__main__':
    main()
