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
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import math
import numpy as np
import random

# Ensure maze_generator is importable from the scripts directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from maze_generator import (
    PACMAN_NAME, GHOST_NAMES, NRF_RADIUS_M,
    world_to_grid, CELL_SIZE, SPAWN_Z, PELLET_Z, cell_center_world,
    generate_map, compute_ghost_starts, N_GHOSTS,
    POWER, PELLET, WALL, EMPTY, grid_to_world
)
from cbba import CBBA_Agent
from allocator import TaskType
from beliefmap import BeliefMap
from pathfinder import astar
from pacman import load_rl_model
import pacman


class GhostNode(Node):

    def __init__(self, name: str, ghost_id: int):
        super().__init__(f'ghost_node_{name}')
        self._name      = name
        self._ghost_id  = ghost_id

        # ── State ────────────────────────────────────────────────────────────
        self._x = 0.0
        self._y = 0.0
        self._z = SPAWN_Z
        self._yaw = 0.0
        self._row = 0
        self._col = 0
        self._pacman_pos: tuple[int, int] | None = None
        self._pacman_powered: bool = False
        self._frame: int = 0
        self._dead = False

        import os
        seed_val = int(os.environ.get('PACMAN_SEED', 42))
        self._global_map, self._player_start = generate_map(seed=seed_val)
        starts = compute_ghost_starts(self._global_map, self._player_start, N_GHOSTS)
        self._start = starts[self._ghost_id]
        
        self._rows, self._cols = self._global_map.shape
        self._visible_cells = set()
        self._pacman_pos_lidar: tuple[int, int] | None = None

        self._personal_map_array = np.full((self._rows, self._cols), -1, dtype=np.int8)
        self.cbba_agent = CBBA_Agent(self._ghost_id)
        self.belief_map = BeliefMap(self._ghost_id, self._personal_map_array, pacman_start=self._player_start)
        self._tail_pacman_remaining = 0
        
        self.last_seen = np.full((self._rows, self._cols), -1, dtype=np.int32)
        self._last_seen = {}
        self._rl_loaded = load_rl_model()
        self.in_fallback_mode = False
        
        self._nav_dir = (0, 0)
        self._target_row, self._target_col = self._start[0], self._start[1]
        self._pending_target = (self._target_row, self._target_col)
        self._arrived = True
        self._centering = False
        self._centering_axis = 'x'

        # NRF24 message tracking and relay
        self._message_queue = []
        self._seen_msg_ids: set = set()
        self._seq = 0
        
        # Shared State
        self._personal_map = {}
        self._last_seen = {}
        self._known_agents = {}
        self._last_heartbeat = {}
        self._last_sync_frame = {}
        self._last_synced_map = {}
        
        self._known_pacman = None
        self._pacman_last_seen = -1
        self._last_lost_pacman = None

        # ── Publishers ───────────────────────────────────────────────────────
        self._cmd_pub  = self.create_publisher(Twist, f'/{name}/cmd_vel', 10)
        self._nrf_pub  = self.create_publisher(String, f'/nrf24/{name}/tx', 20)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Odometry, f'/{name}/odom',  self._odom_cb,  10)
        self.create_subscription(String,   f'/nrf24/{name}/rx', self._nrf_rx_cb, 20)
        self.create_subscription(Bool,     '/pacman_bot/power_state', self._pac_pow_cb, 10)
        self.create_subscription(LaserScan, f'/{name}/scan', self._scan_cb, 10)

        self._set_state = self.create_client(SetEntityState, '/set_entity_state')

        # ── Control loop ─────────────────────────────────────────────────────
        self._timer = self.create_timer(1.0 / 30.0, self._control_loop)  # 30 Hz

        self.get_logger().info(f'Ghost node "{name}" (id={ghost_id}) ready')

    @property
    def grid(self): return self._global_map
    @property
    def row(self): return self._row
    @property
    def col(self): return self._col
    @property
    def gid(self): return self._ghost_id
    @property
    def personal_map(self): return self._personal_map_array
    @property
    def grid(self): return self._global_map
    @property
    def row(self): return self._row
    @property
    def col(self): return self._col
    @property
    def frame(self): return self._frame
    @property
    def known_agents(self): return self._known_agents
    @property
    def known_pacman(self): return self._known_pacman
    @property
    def last_lost_pacman(self): return getattr(self, '_last_lost_pacman', None)
    @property
    def pacman_last_seen(self): return getattr(self, '_pacman_last_seen', -1)
    @property
    def pacman_powered(self): return self._pacman_powered

    @property
    def pacman_power_timer(self): return getattr(self, '_pacman_power_timer', 0)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._z = msg.pose.pose.position.z
        self._row, self._col = world_to_grid(self._x, self._y)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny_cosp, cosy_cosp)

    def _nrf_rx_cb(self, msg: String):
        """Handle incoming NRF24 packet from bridge."""
        try:
            pkt = json.loads(msg.data)
        except Exception:
            return
        self._message_queue.append(pkt)

    def _pac_pow_cb(self, msg: Bool):
        """Receive pacman power-state broadcast (simulates visual color change)."""
        self._true_pacman_powered = msg.data

    def _scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        valid_mask = (ranges >= msg.range_min) & (ranges <= 4.0)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        
        yaw = self._yaw
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        
        wxs = self._x + xs * math.cos(yaw) - ys * math.sin(yaw)
        wys = self._y + xs * math.sin(yaw) + ys * math.cos(yaw)
        
        hit_wxs = wxs[valid_mask]
        hit_wys = wys[valid_mask]
        
        clusters = []
        if len(hit_wxs) > 0:
            current_cluster = [(hit_wxs[0], hit_wys[0])]
            for i in range(1, len(hit_wxs)):
                dist = math.hypot(hit_wxs[i] - hit_wxs[i-1], hit_wys[i] - hit_wys[i-1])
                if dist < 0.2:
                    current_cluster.append((hit_wxs[i], hit_wys[i]))
                else:
                    clusters.append(current_cluster)
                    current_cluster = [(hit_wxs[i], hit_wys[i])]
            clusters.append(current_cluster)
            
        detected_pacman = None
        for c in clusters:
            if not c: continue
            edge_length = math.hypot(c[-1][0] - c[0][0], c[-1][1] - c[0][1])
            if edge_length < 0.20:
                cx = sum(p[0] for p in c) / len(c)
                cy = sum(p[1] for p in c) / len(c)
                cr, cc = world_to_grid(cx, cy)
                
                is_ghost = False
                for gid, pos in self._known_agents.items():
                    if pos != "UNKNOWN":
                        if abs(pos[0] - cr) + abs(pos[1] - cc) <= 1:
                            is_ghost = True
                            break
                if not is_ghost:
                    detected_pacman = (cr, cc)
                    
        self._pacman_pos_lidar = detected_pacman
        
        visible_cells = set()
        step = max(1, len(ranges) // 72)
        for i in range(0, len(ranges), step):
            r = ranges[i]
            if np.isinf(r) or np.isnan(r) or r > 4.0:
                r = 4.0
                
            ray_wx = self._x + r * math.cos(yaw + angles[i])
            ray_wy = self._y + r * math.sin(yaw + angles[i])
            
            dist = math.hypot(ray_wx - self._x, ray_wy - self._y)
            steps = int(dist / 0.1) + 1
            for s in range(steps):
                t = s / steps
                px = self._x + t * (ray_wx - self._x)
                py = self._y + t * (ray_wy - self._y)
                gr, gc = world_to_grid(px, py)
                if 0 <= gr < self._rows and 0 <= gc < self._cols:
                    visible_cells.add((gr, gc))
                    if self._global_map[gr, gc] == 1:
                        break
                        
        self._visible_cells = visible_cells

    # ── NRF24 Data Sharing Pipeline ──────────────────────────────────────────

    def _broadcast(self, diffs, msg_id=None, hop=0):
        if not diffs:
            return
        is_new_msg = msg_id is None
        if is_new_msg:
            msg_id = [int(self._ghost_id), int(self._frame), int(self._seq)]
            self._seq += 1
            cbba_payload = self.cbba_agent.get_consensus_payload()
            belief_payload = self.belief_map.get_payload()
            diffs = list(diffs) + [("cbba", int(self._ghost_id), cbba_payload), ("belief", int(self._ghost_id), belief_payload)]

        key = str(msg_id)
        self._seen_msg_ids.add(key)
        
        pkt = {
            'id': msg_id,
            'diffs': diffs,
            'hop': hop,
        }
        self._nrf_pub.publish(String(data=json.dumps(pkt)))

        if is_new_msg:
            # Check for full sync with in-range agents
            for gid, pos in self._known_agents.items():
                if gid == self._ghost_id:
                    continue
                if pos != "UNKNOWN":
                    gr, gc = pos
                    dist = abs(gr - self._row) + abs(gc - self._col)
                    if dist <= 12: # RADIUS from ghost.py
                        last = self._last_sync_frame.get(gid, -1)
                        if self._frame - last >= 50: # RESYNC_EVERY
                            self._last_sync_frame[gid] = self._frame
                            self._send_full_sync(gid)

    def _send_full_sync(self, target_gid):
        sync_diffs = []
        last_map = self._last_synced_map.get(target_gid, {})
        for k, val in self._personal_map.items():
            if val != -1 and last_map.get(k) != val:
                sync_diffs.append(("cell", int(k[0]), int(k[1]), int(val)))
        
        self._last_synced_map[target_gid] = dict(self._personal_map)
        
        for gid, pos in self._known_agents.items():
            if pos == "UNKNOWN":
                sync_diffs.append(("agent_lost", gid))
            else:
                sync_diffs.append(("agent", gid, pos[0], pos[1]))
                
        for gid, hb_frame in self._last_heartbeat.items():
            frames_ago = self._frame - hb_frame
            sync_diffs.append(("hb_sync", gid, frames_ago))
            
        if self._known_pacman is not None:
            sync_diffs.append(("pacman", self._known_pacman[0], self._known_pacman[1], self._pacman_powered, self._pacman_last_seen))
        elif self._last_lost_pacman is not None and self._pacman_last_seen > -1:
            sync_diffs.append(("pacman_lost", self._last_lost_pacman[0], self._last_lost_pacman[1], self._pacman_last_seen))
            
        if sync_diffs:
            sync_id = ["sync", int(self._ghost_id), int(target_gid), int(self._frame)]
            self._seen_msg_ids.add(str(sync_id))
            
            pkt = {
                'id': sync_id,
                'diffs': sync_diffs,
                'hop': 0,
            }
            self._nrf_pub.publish(String(data=json.dumps(pkt)))

    def _process_messages(self):
        for msg in self._message_queue:
            mid = msg.get('id')
            if mid is None: continue
            key = str(mid)
            if key in self._seen_msg_ids: continue
            self._seen_msg_ids.add(key)
            
            hop = msg.get('hop', 0)
            relay_diffs = []
            
            for diff in msg.get('diffs', []):
                dtype = diff[0]
                if dtype == "cell":
                    _, r, c, val = diff
                    old = self._personal_map.get((r, c), -1)
                    if old != val:
                        if old != -1 and self._last_seen.get((r, c), -999) >= self._frame - 10:
                            continue
                        self._personal_map[(r, c)] = val
                        self._personal_map_array[r, c] = val
                        if val == 1:
                            self.belief_map.update_local_map_cell((r, c), 1)
                        relay_diffs.append(diff)
                elif dtype == "agent":
                    _, gid, r, c = diff
                    if gid == self._ghost_id: continue
                    old = self._known_agents.get(gid)
                    if old != (r, c):
                        self._known_agents[gid] = (r, c)
                        relay_diffs.append(diff)
                elif dtype == "agent_lost":
                    _, gid = diff
                    if gid == self._ghost_id: continue
                    if self._known_agents.get(gid) != "UNKNOWN":
                        self._known_agents[gid] = "UNKNOWN"
                        relay_diffs.append(diff)
                elif dtype == "heartbeat":
                    _, gid, r, c, origin_frame = diff
                    if gid == self._ghost_id: continue
                    existing = self._last_heartbeat.get(gid, -1)
                    if origin_frame > existing:
                        self._last_heartbeat[gid] = origin_frame
                    if r != 0 or c != 0:
                        old = self._known_agents.get(gid)
                        if old != (r, c):
                            self._known_agents[gid] = (r, c)
                            relay_diffs.append(("agent", gid, r, c))
                    relay_diffs.append(diff)
                elif dtype == "hb_sync":
                    _, gid, frames_ago = diff
                    if gid == self._ghost_id: continue
                    reconstructed = self._frame - frames_ago
                    existing = self._last_heartbeat.get(gid, -1)
                    if reconstructed > existing:
                        self._last_heartbeat[gid] = reconstructed
                        relay_diffs.append(diff)
                elif dtype == "pacman":
                    _, r, c, powered, obs_frame = diff
                    if obs_frame > self._pacman_last_seen:
                        self._known_pacman = (r, c)
                        self._pacman_powered = powered
                        self._pacman_last_seen = obs_frame
                        self._last_lost_pacman = None
                        relay_diffs.append(diff)
                elif dtype == "pacman_lost":
                    _, lr, lc, obs_frame = diff
                    if obs_frame > self._pacman_last_seen:
                        if self._known_pacman == (lr, lc):
                            self._known_pacman = None
                        self._last_lost_pacman = (lr, lc)
                        self._pacman_last_seen = obs_frame
                        relay_diffs.append(diff)
                elif dtype in ("cbba", "belief"):
                    _, sender_gid, payload = diff
                    if sender_gid == self._ghost_id: continue
                    if dtype == "cbba":
                        changed = self.cbba_agent.receive_consensus(sender_gid, payload["y"], payload["z"], payload["s"], self._frame)
                        if changed: relay_diffs.append(diff)
                    else:
                        self.belief_map.merge(sender_gid, payload, self._frame)
                        relay_diffs.append(diff)

            if relay_diffs:
                MAX_RELAY_SIZE = 50
                for idx, i in enumerate(range(0, len(relay_diffs), MAX_RELAY_SIZE)):
                    chunk = relay_diffs[i : i + MAX_RELAY_SIZE]
                    if idx == 0:
                        chunk_msg_id = msg["id"]
                    else:
                        chunk_msg_id = list(msg["id"]) if isinstance(msg["id"], (list, tuple)) else [msg["id"]]
                        chunk_msg_id.append(f"chunk_{idx}")
                    self._broadcast(chunk, msg_id=chunk_msg_id, hop=hop+1)
                    
        self._message_queue.clear()
        
        if len(self._seen_msg_ids) > 500:
            to_del = list(self._seen_msg_ids)[:250]
            for k in to_del:
                self._seen_msg_ids.discard(k)

    # ── AI and Movement ──────────────────────────────────────────────────────

    def _choose_next_target(self):
        cr, cc = self._target_row, self._target_col
        
        if self._rl_loaded and pacman.RL_ACTOR is not None and self._frame % 6 == 0:
            from obs import build_spatial, build_vector, build_valid_mask, actions_to_tasks
            import torch
            
            if not hasattr(self, '_recent_nom'):
                self._recent_nom = np.zeros((self._rows, self._cols), dtype=np.float32)
                
            self._recent_nom *= 0.8
            sp = build_spatial(self, self._recent_nom, self._rows, self._cols)
            ve = build_vector(self)
            vm = build_valid_mask(self, self._rows, self._cols)
            
            t_sp = torch.tensor(np.expand_dims(sp, axis=0), device=pacman.RL_DEVICE, dtype=torch.float32)
            t_ve = torch.tensor(np.expand_dims(ve, axis=0), device=pacman.RL_DEVICE, dtype=torch.float32)
            t_vm = torch.tensor(np.expand_dims(vm, axis=0), device=pacman.RL_DEVICE, dtype=torch.bool)
            
            with torch.inference_mode():
                idx, _, scores, _, _ = pacman.RL_ACTOR(t_sp, t_ve, t_vm, K=3)
            
            idx_np = idx.cpu().numpy()[0]
            sc_np = scores.cpu().numpy()[0]
            
            indices = [(int(x // self._cols), int(x % self._cols)) for x in idx_np]
            for r, c in indices:
                if 0 <= r < self._rows and 0 <= c < self._cols:
                    self._recent_nom[r, c] = 1.0
                    
            tasks = actions_to_tasks(self, sc_np, indices, self._frame)
            self.cbba_agent._last_auction = self._frame
            
            h_dists = {}
            self.cbba_agent._task_map.clear()
            if tasks:
                all_targets = [t.target_pos for t in tasks]
                from pathfinder import dijkstra_multi
                h_dists.update(dijkstra_multi(self._global_map, (cr, cc), all_targets))
            self.cbba_agent._phase1(self, tasks, h_dists)

        active_task = self.cbba_agent.step(self, self._frame)
        if active_task and self._pacman_powered and active_task.task_type == TaskType.HUNT:
            active_task = None
            
        if active_task is not None and (cr, cc) == active_task.target_pos:
            key = (int(active_task.task_type), active_task.target_pos, getattr(active_task, 'owner', -1))
            if key in self.cbba_agent.path: self.cbba_agent.path.remove(key)
            if key in self.cbba_agent.bundle: self.cbba_agent.bundle.remove(key)
            active_task = None
            
        moved = False
        dr, dc = 0, 0
        
        if not self._pacman_powered and self._known_pacman:
            pr, pc = self._known_pacman
            if abs(cr - pr) + abs(cc - pc) == 1:
                dr, dc = pr - cr, pc - cc
                moved = True
                self.cbba_agent.bundle.clear()
                self.cbba_agent.path.clear()
                active_task = None
                
                nearby_ghosts = 1
                for _gid, pos in self._known_agents.items():
                    if pos != "UNKNOWN":
                        if abs(pos[0] - cr) + abs(pos[1] - cc) <= 6:
                            nearby_ghosts += 1
                self._tail_pacman_remaining = 2 if nearby_ghosts <= 2 else 0

        if not moved and self._tail_pacman_remaining > 0:
            if self._known_pacman and not self._pacman_powered:
                pr, pc = self._known_pacman
                best_dir = None
                best_dist = abs(cr - pr) + abs(cc - pc)
                for ndr, ndc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = cr + ndr, cc + ndc
                    if 0 <= nr < self._rows and 0 <= nc < self._cols and self._global_map[nr, nc] != 1:
                        d = abs(nr - pr) + abs(nc - pc)
                        if d < best_dist:
                            best_dist = d
                            best_dir = (ndr, ndc)
                if best_dir is not None:
                    dr, dc = best_dir
                    moved = True
                    self._tail_pacman_remaining -= 1
                else:
                    self._tail_pacman_remaining = 0
            else:
                self._tail_pacman_remaining = 0
                
        if not moved and active_task is not None:
            target = active_task.target_pos
            full_path = astar(self._global_map, (cr, cc), target)
            if len(full_path) >= 2:
                nxt = full_path[1]
                dr, dc = nxt[0] - cr, nxt[1] - cc
                moved = True

        if not moved:
            pac_cell = self._known_pacman if (self._pacman_powered and self._known_pacman) else None
            options = []
            for ndr, ndc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = cr + ndr, cc + ndc
                if 0 <= nr < self._rows and 0 <= nc < self._cols and self._global_map[nr, nc] != 1 and (nr, nc) != pac_cell:
                    options.append((ndr, ndc))
            if options:
                if self._nav_dir in options and random.random() < 0.70:
                    dr, dc = self._nav_dir
                else:
                    dr, dc = random.choice(options)
                moved = True
            else:
                dr, dc = -self._nav_dir[0], -self._nav_dir[1]
                moved = True

        self._nav_dir = (dr, dc)
        self._target_row = cr + dr
        self._target_col = cc + dc

    def _compute_cmd(self, tx: float, ty: float) -> Twist:
        cmd = Twist()
        err_x = tx - self._x
        err_y = ty - self._y
        dist = math.hypot(err_x, err_y)

        if dist < 0.01:
            cmd.angular.z = -15.0 * self._yaw
            return cmd

        dr, dc = self._nav_dir
        vx = 0.0
        vy = 0.0

        if getattr(self, '_centering', False):
            p_gain = 1.0 / 0.25
            if self._centering_axis == 'x':
                vx_raw = err_x * p_gain
                vx = math.copysign(max(0.35, min(1.0, abs(vx_raw))), vx_raw)
            else:
                vy_raw = err_y * p_gain
                vy = math.copysign(max(0.35, min(1.0, abs(vy_raw))), vy_raw)
        else:
            speed = 1.0
            if dc != 0:
                vx = math.copysign(speed, err_x)
            elif dr != 0:
                vy = math.copysign(speed, err_y)

        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        cmd.linear.x = vx * cos_y + vy * sin_y
        cmd.linear.y = -vx * sin_y + vy * cos_y
        cmd.angular.z = -15.0 * self._yaw
        return cmd

    def _teleport_self(self, x: float, y: float, z: float):
        if not self._set_state.service_is_ready():
            return
        req = SetEntityState.Request()
        req.state.name = f"ghost_{self._ghost_id}"
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = z
        req.state.pose.orientation.w = 1.0
        self._set_state.call_async(req)

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_loop(self):
        self._frame += 1
        
        # Convert Power Pellet to Pellet
        if int(self._global_map[self._row, self._col]) == POWER:
            self._global_map[self._row, self._col] = PELLET
            self._personal_map[(self._row, self._col)] = PELLET
            self._personal_map_array[self._row, self._col] = PELLET
            self.belief_map.update_local_map_cell((self._row, self._col), 1)
            
            if not hasattr(self, '_pending_conversions'):
                self._pending_conversions = []
            self._pending_conversions.append(("cell", self._row, self._col, PELLET))
            
            # Gazebo swaps
            if self._set_state.service_is_ready():
                wx, wy, _ = grid_to_world(self._row, self._col)
                # Move power down
                req1 = SetEntityState.Request()
                req1.state.name = f'pellet_field::power_{self._row}_{self._col}'
                req1.state.pose.position.x = wx
                req1.state.pose.position.y = wy
                req1.state.pose.position.z = -2.0
                req1.state.pose.orientation.w = 1.0
                self._set_state.call_async(req1)
                
                # Move hidden white pellet up
                req2 = SetEntityState.Request()
                req2.state.name = f'pellet_field::pellet_{self._row}_{self._col}'
                req2.state.pose.position.x = wx
                req2.state.pose.position.y = wy
                req2.state.pose.position.z = PELLET_Z
                req2.state.pose.orientation.w = 1.0
                self._set_state.call_async(req2)
        
        # Death handling (Ghost sent below ground to -2.0)
        if self._dead:
            if self._z >= -1.0:
                self._dead = False
                self._row, self._col = self._start
                self._target_row, self._target_col = self._start
                self._arrived = True
                self._centering = False
            else:
                self._cmd_pub.publish(Twist())
                return
        elif self._z < -1.0:
            self._dead = True
            self._cmd_pub.publish(Twist())
            self.cbba_agent.bundle.clear()
            self.cbba_agent.path.clear()
            return

        self._process_messages()
        
        if getattr(self, '_pacman_powered', False):
            self._pacman_power_timer = getattr(self, '_pacman_power_timer', 0) - 1
            if self._pacman_power_timer <= 0:
                self._pacman_powered = False
        
        # Belief Map Updates
        self.belief_map.diffuse((self._row, self._col))
        pacman_in_los = (self._pacman_pos_lidar is not None)
        if pacman_in_los:
            pr, pc = self._pacman_pos_lidar
            
            is_powered = getattr(self, '_true_pacman_powered', False)
            if is_powered and not self._pacman_powered:
                self._pacman_power_timer = 120
            self._pacman_powered = is_powered
            if not is_powered:
                self._pacman_power_timer = 0
            
            pac_dir = (0, 0)
            if hasattr(self, '_prev_pac_row') and self._prev_pac_row >= 0:
                pac_dir = (pr - self._prev_pac_row, pc - self._prev_pac_col)
            self.belief_map.observe((pr, pc), pac_dir)
            self._prev_pac_row, self._prev_pac_col = pr, pc
        elif self._last_lost_pacman is not None and self._pacman_last_seen == self._frame:
            self.belief_map.observe_lost(self._last_lost_pacman)
            
        pac_pos = self._pacman_pos_lidar if pacman_in_los else None
        self.belief_map.observe_clear(self._visible_cells, pac_pos)
        self.belief_map.update_safety_map(self._known_agents, self._frame, powered=self._pacman_powered)

        # Map generation
        diffs = []
        if hasattr(self, '_pending_conversions'):
            for diff in self._pending_conversions:
                diffs.append(diff)
            self._pending_conversions.clear()
            
        for (r, c) in self._visible_cells:
            self.last_seen[r, c] = self._frame
            self._last_seen[(r, c)] = self._frame
            val = int(self._global_map[r, c])
            old = self._personal_map.get((r, c), -1)
            if old != val:
                self._personal_map[(r, c)] = val
                self._personal_map_array[r, c] = val
                if val == 1:
                    self.belief_map.update_local_map_cell((r, c), 1)
                diffs.append(("cell", r, c, val))

        # Movement Controller
        tx, ty = cell_center_world(self._target_row, self._target_col)
        dist = math.hypot(self._x - tx, self._y - ty)

        if dist < 0.25:
            if not self._arrived:
                self._arrived = True
                old_dir = self._nav_dir
                intersect_r = self._target_row
                intersect_c = self._target_col
                
                self._choose_next_target()

                if self._nav_dir != old_dir:
                    self._pending_target = (self._target_row, self._target_col)
                    self._target_row = intersect_r
                    self._target_col = intersect_c
                    self._centering = True
                    self._centering_axis = 'x' if old_dir[1] != 0 else 'y'

                tx, ty = cell_center_world(self._target_row, self._target_col)
        else:
            self._arrived = False

        if getattr(self, '_centering', False):
            cx, cy = cell_center_world(self._target_row, self._target_col)
            err_c = (self._x - cx) if self._centering_axis == 'x' else (self._y - cy)
            if abs(err_c) < 0.01:
                self._centering = False
                self._target_row = self._pending_target[0]
                self._target_col = self._pending_target[1]
                self._arrived = False
                tx, ty = cell_center_world(self._target_row, self._target_col)

        self._cmd_pub.publish(self._compute_cmd(tx, ty))

        # ── NRF24 heartbeat and basic pacman sighting ────────────────────────
        if self._frame % 5 == 0:
            diffs.append(('heartbeat', int(self._ghost_id), int(self._row), int(self._col), int(self._frame)))
            
            # Pacman sighting based on LIDAR elimination strategy
            if self._pacman_pos_lidar:
                self._known_pacman = self._pacman_pos_lidar
                self._pacman_last_seen = self._frame
                diffs.append(("pacman", self._pacman_pos_lidar[0], self._pacman_pos_lidar[1], self._pacman_powered, self._frame))
            
            self._broadcast(diffs)

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
