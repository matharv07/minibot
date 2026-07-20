#!/usr/bin/env python3
"""
pacman_node.py — Pac-Man bot ROS 2 node (Gazebo 11).

Navigation
──────────
• Grid-locked: bot moves cell-centre → cell-centre.
• Direction chosen by an Adam-momentum potential-field navigator
  (ported directly from pacman.py Player.update() AUTO_MODE logic).
• pathfinder.py scipy-dijkstra used for ghost repulsion maps.
• Fallback BFS pellet-chase if pathfinder unavailable.

Motion: planar_move plugin via /pacman_bot/cmd_vel (linear.x/y only, angular.z=0).
"""

import json
import math
import sys
import os
import random
import numpy as np

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
    generate_map,
)

# ── Motion constants ──────────────────────────────────────────────────────────
BOT_SPEED      = 2.0           # m/s
ARRIVE_DIST    = 0.25          # m — early lookahead for smooth braking before turns
CELL_SNAP_DIST = 0.03          # m — dead-band: zero cmd if closer than this
CROSS_KP       = 6.0           # cross-axis P gain (gentle centering)
DECEL_DIST     = 0.10          # m — start decelerating this far from target
DECEL_MIN_SPD  = 0.8           # m/s — minimum speed during deceleration

# ── Power pellet ──────────────────────────────────────────────────────────────
POWER_TICKS   = int((40 * CELL_SIZE / BOT_SPEED) * 30.0) # 40 cells * (time per cell at BOT_SPEED) at 30 Hz
PELLET_RADIUS = 0.075       # 7.5 cm (approx bot physical radius)

# ── NRF ───────────────────────────────────────────────────────────────────────
HEARTBEAT_EVERY = 15        # 0.5 seconds at 30 Hz
GHOST_TIMEOUT   = 180       # 6 seconds before ghost position is stale

# ── Directions ────────────────────────────────────────────────────────────────
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # UP DOWN LEFT RIGHT


# ─────────────────────────────────────────────────────────────────────────────
# Standalone BFS helpers (no pacman.py import needed)
# ─────────────────────────────────────────────────────────────────────────────

def _bfs_from_sources(grid: np.ndarray, sources) -> np.ndarray:
    """Multi-source BFS. Returns distance map (inf for walls / unreachable)."""
    dist = np.full(grid.shape, np.inf)
    active = np.zeros(grid.shape, dtype=bool)
    wall_mask = (grid == WALL)
    for r, c in sources:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            dist[r, c] = 0
            active[r, c] = True
    d = 0
    while active.any():
        d += 1
        new = (np.roll(active, -1, 0) | np.roll(active, 1, 0) |
               np.roll(active, -1, 1) | np.roll(active, 1, 1))
        new &= ~wall_mask & np.isinf(dist)
        dist[new] = d
        active = new
    return dist


def _pellet_bfs(grid: np.ndarray) -> np.ndarray:
    sources = list(zip(*np.where(np.isin(grid, [PELLET, POWER]))))
    if not sources:
        return np.full(grid.shape, np.inf)
    return _bfs_from_sources(grid, sources)


def _ghost_bfs(grid: np.ndarray, ghost_cells: list) -> np.ndarray:
    if not ghost_cells:
        return np.full(grid.shape, np.inf)
    return _bfs_from_sources(grid, ghost_cells)


# ─────────────────────────────────────────────────────────────────────────────

class PacmanNode(Node):

    def __init__(self):
        super().__init__('pacman_node')
        self._cbg = ReentrantCallbackGroup()

        # ── Map ──────────────────────────────────────────────────────────────
        seed_val = int(os.environ.get('PACMAN_SEED', 42))
        self._grid, self._start = generate_map(seed=seed_val)
        self._rows, self._cols  = self._grid.shape
        self._consumed: set     = set()

        # ── Build pathfinder graph (scipy dijkstra) ───────────────────────────
        self._pf        = None
        self._pf_graph  = None   # (csr, open_cells, cell_to_idx)
        try:
            import pathfinder as _pf_mod
            _pf_mod.build_scipy_graph(self._grid)
            self._pf       = _pf_mod
            self._pf_graph = _pf_mod.get_scipy_graph(self._grid)
            self.get_logger().info('pathfinder graph built OK')
        except Exception as e:
            self.get_logger().warn(f'pathfinder unavailable ({e}) — using BFS navigation')

        # ── Physical state ────────────────────────────────────────────────────
        self._row, self._col = self._start
        self._x, self._y     = cell_center_world(self._row, self._col)
        self._z              = SPAWN_Z
        self._yaw            = 0.0
        self._q              = None

        # ── Game state ────────────────────────────────────────────────────────
        self._score         = 0
        self._powered       = False
        self._power_timer   = 0
        self._dead          = False
        self._dead_timer    = 0
        self._pellets_eaten = 0
        self._power_eaten   = 0
        self._ghosts_eaten  = 0
        self._ghosts_eaten_this_power = 0
        self._last_points   = 0
        self._power_neutralised = 0

        # ── Ghost knowledge (must be before first _choose_next_target call) ───
        self._ghost_pos:       dict = {i: None for i in range(N_GHOSTS)}
        self._ghost_last_seen: dict = {}

        # ── Adam navigator state (mirrors Player in pacman.py) ────────────────
        self._nav_dir    = (0, 1)   # initial heading: RIGHT
        self._m_row      = 0.0
        self._m_col      = 0.0
        self._v_row      = 0.0
        self._v_col      = 0.0
        self._adam_t     = 0
        self._beta1      = 0.9
        self._beta2      = 0.999
        self._eps        = 1e-8
        self._macro_mode = False    # True → follow pellet gradient directly

        # ── Navigation target ────────────────────────────────────────────────
        self._target_row = self._row
        self._target_col = self._col
        self._arrived    = False     # one-shot flag: True = at target, need new
        self._centering  = False     # True = homing to cell centre before turning
        self._centering_axis = 'x'
        self._pending_target = (self._row, self._col)

        # ── NRF dedup ─────────────────────────────────────────────────────────
        self._seen_ids: set = set()
        self._nrf_seq       = 0
        self._tick          = 0

        # ── Publishers ───────────────────────────────────────────────────────
        self._cmd_pub   = self.create_publisher(Twist,  f'/{PACMAN_NAME}/cmd_vel', 10)
        self._nrf_pub   = self.create_publisher(String, f'/nrf24/{PACMAN_NAME}/tx', 20)
        self._loc_pub   = self.create_publisher(Point,  '/pacman_bot/location',    10)
        self._power_pub = self.create_publisher(Bool,   '/pacman_bot/power_state', 10)
        self._stats_pub = self.create_publisher(String, '/pacman_bot/stats',       10)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Odometry, f'/{PACMAN_NAME}/odom',
                                 self._odom_cb, 10, callback_group=self._cbg)
        self.create_subscription(String, f'/nrf24/{PACMAN_NAME}/rx',
                                 self._nrf_rx_cb, 20, callback_group=self._cbg)

        # ── Gazebo service ────────────────────────────────────────────────────
        self._set_state = self.create_client(SetEntityState, '/set_entity_state')

        # ── 30 Hz control loop ────────────────────────────────────────────────
        self.create_timer(1.0 / 30.0, self._control_loop, callback_group=self._cbg)
        self.get_logger().info(f'pacman_node ready — {self._rows}×{self._cols} grid, speed={BOT_SPEED} m/s')

    # ── Subscribers ──────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._z = msg.pose.pose.position.z
        self._q = msg.pose.pose.orientation
        self._row, self._col = world_to_grid(self._x, self._y)
        
        q = self._q
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny_cosp, cosy_cosp)

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
            if dtype in ('heartbeat', 'agent'):
                _, gid, r, c = diff[:4]
                if 0 <= gid < N_GHOSTS:
                    self._ghost_pos[gid]       = (int(r), int(c))
                    self._ghost_last_seen[gid] = self._tick

    # ── Pellet consumption ────────────────────────────────────────────────────

    def _try_consume(self, row: int, col: int):
        if (row, col) in self._consumed:
            return
        cell = self._grid[row, col]
        if cell not in (PELLET, POWER):
            return
        self._consumed.add((row, col))
        self._grid[row, col] = EMPTY

        if cell == PELLET:
            self._score        += 10
            self._pellets_eaten += 1
            self._last_points   = 10
        else:
            self._score        += 50
            self._power_eaten  += 1
            self._last_points   = 50
            self._powered       = True
            self._power_timer   = POWER_TICKS
            self._ghosts_eaten_this_power = 0
            # Invalidate pathfinder cache (grid changed significantly)
            if self._pf is not None:
                try:
                    self._pf.build_scipy_graph(self._grid)
                    self._pf_graph = self._pf.get_scipy_graph(self._grid)
                except Exception:
                    pass

        # Flush momentum info to prevent rubber-banding artifacts (matches pacman.py)
        self._m_row = self._m_col = 0.0
        self._v_row = self._v_col = 0.0
        self._adam_t = 0

        wx, wy, _ = grid_to_world(row, col)
        
        entities_to_hide = []
        if cell == POWER:
            entities_to_hide.append(f'pellet_field::power_{row}_{col}')
            entities_to_hide.append(f'pellet_field::pellet_{row}_{col}')
        else:
            entities_to_hide.append(f'pellet_field::pellet_{row}_{col}')
            
        if self._set_state.service_is_ready():
            for entity in entities_to_hide:
                req   = SetEntityState.Request()
                state = EntityState()
                state.name = entity
                state.pose.position.x = wx
                state.pose.position.y = wy
                state.pose.position.z = -2.0
                state.pose.orientation.w = 1.0
                req.state = state
                future = self._set_state.call_async(req)
                future.add_done_callback(
                    lambda f, e=entity: self._on_set_state_done(f, e)
                )
        else:
            self.get_logger().warn(f'SetEntityState service NOT ready — cannot hide {entity}')

    def _on_set_state_done(self, future, entity_name):
        """Log result of SetEntityState call for debugging pellet removal."""
        try:
            res = future.result()
            if res is not None and not res.success:
                self.get_logger().error(
                    f'SetEntityState FAILED for [{entity_name}] — entity not found in Gazebo')
        except Exception as exc:
            self.get_logger().error(
                f'SetEntityState exception for [{entity_name}]: {exc}')

    # ── Adam potential-field navigator ────────────────────────────────────────

    def _ghost_maps_scipy(self) -> list:
        """Use scipy dijkstra to build per-ghost distance maps. Returns list of np arrays."""
        if self._pf is None or self._pf_graph is None:
            return []
        graph, open_cells, cell_to_idx = self._pf_graph
        g_indices = []
        for gid, pos in self._ghost_pos.items():
            if pos is None:
                continue
            if self._tick - self._ghost_last_seen.get(gid, 0) > GHOST_TIMEOUT:
                continue
            if pos in cell_to_idx:
                g_indices.append(cell_to_idx[pos])
        if not g_indices:
            return []
        try:
            dm = self._pf.scipy_dijkstra(
                csgraph=graph, directed=False, indices=g_indices)
            if dm.ndim == 1:
                dm = dm[np.newaxis, :]
            r_arr = np.array([r for r, c in open_cells])
            c_arr = np.array([c for r, c in open_cells])
            maps  = []
            for i in range(len(g_indices)):
                g_map = np.full((self._rows, self._cols), np.inf)
                g_map[r_arr, c_arr] = dm[i]
                maps.append(g_map)
            return maps
        except Exception:
            return []

    def _evaluate(self, r: int, c: int,
                  ghost_maps: list, pellet_map: np.ndarray) -> float:
        """Potential at cell (r,c). Lower = better. Mirrors pacman.py._evaluate_potential."""
        if not (0 <= r < self._rows and 0 <= c < self._cols):
            return 9999.0
        if self._grid[r, c] == WALL:
            return 9999.0
        g_dists = [float(gm[r, c]) for gm in ghost_maps
                   if not math.isinf(gm[r, c]) and not math.isnan(gm[r, c])]
        if self._powered:
            # Chase ghosts
            return -min(g_dists) * 15.0 if g_dists else float(pellet_map[r, c])
        ghost_rep = 0.0
        for d in g_dists:
            if d <= 4:
                ghost_rep += 200.0 / (d + 0.1)
            elif d <= 8:
                ghost_rep += 40.0 / (d + 0.1)
        p = float(pellet_map[r, c])
        ct = self._grid[r, c]
        weight = 5.0 if ct == POWER else 1.2
        pellet_attr = p * weight if math.isfinite(p) and not math.isnan(p) else 0.0
        return ghost_rep + pellet_attr

    def _choose_next_target(self):
        """
        Adam-momentum potential-field step (mirrors pacman.py Player.update AUTO_MODE).
        Sets self._target_row / _target_col to the best adjacent non-wall cell.
        """
        cr, cc = self._target_row, self._target_col  # Plan from the cell we are arriving at

        # Ghost maps
        ghost_maps = self._ghost_maps_scipy()
        if not ghost_maps:
            # Fallback: single BFS map from a stale ghost position or (0,0)
            active_ghosts = [pos for gid, pos in self._ghost_pos.items()
                             if pos is not None and
                             self._tick - self._ghost_last_seen.get(gid, 0) <= GHOST_TIMEOUT]
            ghost_maps = [_ghost_bfs(self._grid, active_ghosts)] if active_ghosts else []

        pellet_map = _pellet_bfs(self._grid)
        p_here     = float(pellet_map[cr, cc]) if not np.isinf(pellet_map[cr, cc]) else 999.0

        # Decide macro vs micro routing (mirrors pacman.py)
        min_ghost_dist = min(
            (float(gm[cr, cc]) for gm in ghost_maps if not math.isinf(gm[cr, cc])),
            default=float('inf'))
        if self._macro_mode:
            if p_here <= 1 or min_ghost_dist <= 4:
                self._macro_mode = False
        else:
            if p_here > 3 and min_ghost_dist > 6:
                self._macro_mode = True

        def can_move(dr, dc):
            nr, nc = cr + dr, cc + dc
            return (0 <= nr < self._rows and 0 <= nc < self._cols
                    and self._grid[nr, nc] != WALL)

        if self._macro_mode and math.isfinite(p_here):
            # Pure pellet-gradient descent — no Adam
            best_dir  = self._nav_dir
            best_dist = p_here
            for dr, dc in _DIRS:
                if not can_move(dr, dc):
                    continue
                nr, nc = cr + dr, cc + dc
                d = float(pellet_map[nr, nc])
                if d < best_dist:
                    best_dist = d
                    best_dir  = (dr, dc)
            self._nav_dir = best_dir
            # Reset Adam on macro mode
            self._m_row = self._m_col = 0.0
            self._v_row = self._v_col = 0.0
            self._adam_t = 0
        else:
            # Adam gradient step
            val_up    = self._evaluate(cr - 1, cc,     ghost_maps, pellet_map)
            val_down  = self._evaluate(cr + 1, cc,     ghost_maps, pellet_map)
            val_left  = self._evaluate(cr,     cc - 1, ghost_maps, pellet_map)
            val_right = self._evaluate(cr,     cc + 1, ghost_maps, pellet_map)

            grad_row = val_up   - val_down
            grad_col = val_left - val_right

            self._adam_t  += 1
            self._m_row = self._beta1 * self._m_row + (1 - self._beta1) * grad_row
            self._m_col = self._beta1 * self._m_col + (1 - self._beta1) * grad_col
            self._v_row = self._beta2 * self._v_row + (1 - self._beta2) * grad_row ** 2
            self._v_col = self._beta2 * self._v_col + (1 - self._beta2) * grad_col ** 2

            t = max(1, self._adam_t)
            mhr = self._m_row / (1 - self._beta1 ** t)
            mhc = self._m_col / (1 - self._beta1 ** t)
            vhr = self._v_row / (1 - self._beta2 ** t)
            vhc = self._v_col / (1 - self._beta2 ** t)
            step_r = mhr / (math.sqrt(max(0.0, vhr)) + self._eps)
            step_c = mhc / (math.sqrt(max(0.0, vhc)) + self._eps)
            if not math.isfinite(step_r): step_r = 0.0
            if not math.isfinite(step_c): step_c = 0.0

            scored, fallback = [], []
            for dr, dc in _DIRS:
                if not can_move(dr, dc):
                    continue
                nr, nc = cr + dr, cc + dc
                score = dr * step_r + dc * step_c
                if (dr, dc) == self._nav_dir:
                    score += 0.8    # heading retention
                if (dr, dc) == (-self._nav_dir[0], -self._nav_dir[1]):
                    score -= 2.2    # U-turn penalty
                # Lethal threat check
                lethal = (not self._powered and
                          any(gm[nr, nc] <= 1 for gm in ghost_maps))
                (fallback if lethal else scored).append((score, (dr, dc)))

            moves = scored if scored else fallback
            if not moves:
                # No valid moves at all — stay put
                return
            moves.sort(key=lambda x: x[0], reverse=True)
            rand = random.random()
            if rand < 0.05 and len(moves) > 2:
                chosen = moves[2][1]
            elif rand < 0.18 and len(moves) > 1:
                chosen = moves[1][1]
            else:
                chosen = moves[0][1]
            self._nav_dir = chosen

        dr, dc = self._nav_dir
        nr, nc = cr + dr, cc + dc
        if (0 <= nr < self._rows and 0 <= nc < self._cols
                and self._grid[nr, nc] != WALL):
            self._target_row = nr
            self._target_col = nc
        else:
            # Heading blocked — pick any valid neighbour
            for dr2, dc2 in _DIRS:
                nr2, nc2 = cr + dr2, cc + dc2
                if (0 <= nr2 < self._rows and 0 <= nc2 < self._cols
                        and self._grid[nr2, nc2] != WALL):
                    self._target_row = nr2
                    self._target_col = nc2
                    self._nav_dir    = (dr2, dc2)
                    # Reset Adam on forced redirect
                    self._m_row = self._m_col = 0.0
                    self._v_row = self._v_col = 0.0
                    self._adam_t = 0
                    break

    # ── Main control loop (10 Hz) ─────────────────────────────────────────────

    def _control_loop(self):
        self._tick += 1

        # Power timer
        if self._powered:
            self._power_timer -= 1
            if self._power_timer <= 0:
                self._powered = False

        # Death recovery
        if self._dead:
            self._dead_timer -= 1
            if self._dead_timer <= 0:
                self._dead = False
                self._row, self._col = self._start
                self._target_row, self._target_col = self._start
                self._arrived   = False
                self._centering = False
                self._centering_axis = 'x'
                self._powered   = False
                self._power_timer = 0
                self._m_row = self._m_col = 0.0
                self._v_row = self._v_col = 0.0
                self._adam_t = 0
                self._teleport_self(*grid_to_world(self._row, self._col), force_z=SPAWN_Z)
            self._cmd_pub.publish(Twist())
            return

        # Pellet consumption at current physical cell
        cx, cy = cell_center_world(self._row, self._col)
        if math.hypot(self._x - cx, self._y - cy) < PELLET_RADIUS:
            self._try_consume(self._row, self._col)

        # Check ghost collisions
        for gid, pos in self._ghost_pos.items():
            if pos is None:
                continue
            if self._tick - self._ghost_last_seen.get(gid, 0) > GHOST_TIMEOUT:
                continue
            gr, gc = pos
            
            if self._grid[gr, gc] == POWER:
                self._grid[gr, gc] = PELLET
                self._power_neutralised += 1
                
                wx, wy, _ = grid_to_world(gr, gc)
                if self._set_state.service_is_ready():
                    req   = SetEntityState.Request()
                    state = EntityState()
                    state.name = f'pellet_field::power_{gr}_{gc}'
                    state.pose.position.x = wx
                    state.pose.position.y = wy
                    state.pose.position.z = -2.0
                    state.pose.orientation.w = 1.0
                    req.state = state
                    self._set_state.call_async(req)
                    
            if self._row == gr and self._col == gc:
                if self._powered:
                    points = 200 * (2 ** self._ghosts_eaten_this_power)
                    self._score += points
                    self._last_points = points
                    self._ghosts_eaten += 1
                    self._ghosts_eaten_this_power += 1
                    self.get_logger().info(f'ATE GHOST {gid}! +{points} points. Score: {self._score}')
                    self._ghost_pos[gid] = None
                    if self._set_state.service_is_ready():
                        req = SetEntityState.Request()
                        req.state.name = GHOST_NAMES[gid]
                        sx, sy, _ = grid_to_world(self._start[0], self._start[1])
                        req.state.pose.position.x = sx
                        req.state.pose.position.y = sy
                        req.state.pose.position.z = 5.0
                        self._set_state.call_async(req)
                else:
                    if not self._dead:
                        self.get_logger().info(f'DIED to ghost {gid}!')
                        self._dead = True
                        self._dead_timer = 60

        # Arrival → choose next target (one-shot: fires once per cell)
        tx, ty = cell_center_world(self._target_row, self._target_col)
        dist   = math.hypot(self._x - tx, self._y - ty)

        if dist < ARRIVE_DIST:
            if not self._arrived:
                self._arrived = True
                old_dir = self._nav_dir
                intersect_r = self._target_row
                intersect_c = self._target_col
                self._choose_next_target()

                # Direction changed → center on intersection cell before turning
                if self._nav_dir != old_dir:
                    self._pending_target = (self._target_row, self._target_col)
                    self._target_row = intersect_r
                    self._target_col = intersect_c
                    self._centering  = True
                    self._centering_axis = 'x' if old_dir[1] != 0 else 'y'

                tx, ty = cell_center_world(self._target_row, self._target_col)
        else:
            self._arrived = False

        # Centering complete → proceed to the real next-cell target
        if self._centering:
            cx, cy = cell_center_world(self._target_row, self._target_col)
            err_c = (self._x - cx) if self._centering_axis == 'x' else (self._y - cy)
            if abs(err_c) < CELL_SNAP_DIST:
                self._centering  = False
                self._teleport_self(cx, cy)
                self._target_row = self._pending_target[0]
                self._target_col = self._pending_target[1]
                self._arrived    = False
                tx, ty = cell_center_world(self._target_row, self._target_col)

        # Motion command
        self._cmd_pub.publish(self._compute_cmd(tx, ty))

        # Broadcasts
        loc = Point()
        loc.x, loc.y = float(self._row), float(self._col)
        self._loc_pub.publish(loc)
        self._power_pub.publish(Bool(data=self._powered))

        # Stats (every 9 ticks ≈ 3.3 Hz)
        if self._tick % 9 == 0:
            pl = int(np.sum(np.isin(self._grid, [PELLET, POWER])))
            self._stats_pub.publish(String(data=json.dumps({
                'tick': int(self._tick), 'score': int(self._score),
                'powered': self._powered, 'power_timer': int(self._power_timer),
                'power_timer_max': int(POWER_TICKS),
                'row': int(self._row), 'col': int(self._col), 'pellets_left': pl,
                'pellets_eaten': int(self._pellets_eaten),
                'power_eaten': int(self._power_eaten),
                'power_neutralised': int(self._power_neutralised),
                'ghosts_eaten': int(self._ghosts_eaten),
                'speed': BOT_SPEED,
                'target': [int(self._target_row), int(self._target_col)],
                'last_points': int(self._last_points),
            })))
            self._last_points = 0

        # NRF heartbeat
        if self._tick % HEARTBEAT_EVERY == 0:
            pkt = {
                'id':    ['pacman', int(self._tick), int(self._nrf_seq)],
                'diffs': [('heartbeat', -1, int(self._row), int(self._col), int(self._tick))],
                'hop':   0,
            }
            self._nrf_seq += 1
            self._nrf_pub.publish(String(data=json.dumps(pkt)))

        # Periodic log (every 150 ticks = 5s)
        if self._tick % 150 == 0:
            pl = int(np.sum(np.isin(self._grid, [PELLET, POWER])))
            self.get_logger().info(
                f'score={self._score} powered={self._powered}({self._power_timer}t) ghosts_eaten={self._ghosts_eaten} '
                f'cell=({self._row},{self._col})→({self._target_row},{self._target_col}) '
                f'pellets_left={pl} dist={dist:.3f}m macro={self._macro_mode}'
            )

    # ── Motion controller (straight-line only) ────────────────────────────────

    def _compute_cmd(self, tx: float, ty: float) -> Twist:
        """Strictly orthogonal warehouse-bot movement."""
        cmd   = Twist()
        err_x = tx - self._x
        err_y = ty - self._y
        dist  = math.hypot(err_x, err_y)

        if dist < CELL_SNAP_DIST:
            cmd.angular.z = -15.0 * self._yaw
            return cmd

        dr, dc = self._nav_dir

        vx = 0.0
        vy = 0.0

        if self._centering:
            # Centering phase: just move on the primary axis to prevent diagonal movement
            p_gain = BOT_SPEED / ARRIVE_DIST
            if self._centering_axis == 'x':
                vx_raw = err_x * p_gain
                vx = math.copysign(max(DECEL_MIN_SPD, min(BOT_SPEED, abs(vx_raw))), vx_raw)
                vy = 0.0
            else:
                vy_raw = err_y * p_gain
                vx = 0.0
                vy = math.copysign(max(DECEL_MIN_SPD, min(BOT_SPEED, abs(vy_raw))), vy_raw)
        else:
            # Regular movement along grid axes
            speed = BOT_SPEED
            
            if dc != 0:
                vx = math.copysign(speed, err_x)
                vy = 0.0
            elif dr != 0:
                vy = math.copysign(speed, err_y)
                vx = 0.0

        # World → body frame (yaw held ≈ 0 by strong P-lock below)
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        cmd.linear.x =  vx * cos_y + vy * sin_y
        cmd.linear.y = -vx * sin_y + vy * cos_y

        # Strong yaw lock to 0
        cmd.angular.z = -15.0 * self._yaw

        return cmd

    def _teleport_self(self, x: float, y: float, force_z: float = None):
        if not self._set_state.service_is_ready():
            return
        req   = SetEntityState.Request()
        state = EntityState()
        state.name = PACMAN_NAME
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = force_z if force_z is not None else self._z
        if self._q is not None and force_z is None:
            state.pose.orientation = self._q
        else:
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
