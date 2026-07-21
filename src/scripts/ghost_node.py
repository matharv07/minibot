#!/usr/bin/env python3
import json
import sys
import os
import math
import numpy as np
import random
import torch

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetEntityState

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PACMANBOT_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'pacmanbot')
if _PACMANBOT_DIR not in sys.path:
    sys.path.insert(0, _PACMANBOT_DIR)

from maze_generator import world_to_grid, cell_center_world, SPAWN_Z
from pacman import generate_map, WALL, PELLET, POWER, EMPTY
from beliefmap import BeliefMap
from cbba import CBBA_Agent
from obs import build_spatial, build_vector, build_valid_mask, actions_to_tasks, MAX_GHOSTS, UNKNOWN
from allocator import generate_tasks, TaskType
from net import GhostActor
import pathfinder

BOT_SPEED = 1.0
ARRIVE_DIST = 0.25
CELL_SNAP_DIST = 0.03
CROSS_KP = 6.0
DECEL_MIN_SPD = 0.8
AUCTION_EVERY = 6

HEARTBEAT_EVERY   = 5
HEARTBEAT_TIMEOUT = 25
RESYNC_EVERY      = 50
MEMORY_FRAMES     = 10
OSCILLATION_WINDOW = 8

RAY_COUNT         = 360
MAX_RAY_DIST      = 10
_ANGLES = np.radians(np.arange(RAY_COUNT))
_DX = np.cos(_ANGLES) * 0.5
_DY = np.sin(_ANGLES) * 0.5

RL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import glob
ckpts = glob.glob(os.path.join(_PACMANBOT_DIR, "checkpoints/ckpt_*.pt"))
latest = max(ckpts, key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
RL_ACTOR = GhostActor().to(RL_DEVICE)
checkpoint = torch.load(latest, map_location=RL_DEVICE, weights_only=False)
RL_ACTOR.load_state_dict(checkpoint["actor"])
RL_ACTOR.eval()

class GhostNode(Node):

    def __init__(self, name: str, ghost_id: int):
        super().__init__(f'ghost_node_{name}')
        self._name      = name
        self.gid        = ghost_id

        # ── Physical State ───────────────────────────────────────────────────
        self._x = 0.0
        self._y = 0.0
        self._z = SPAWN_Z
        self._q = None
        self.row = 0
        self.col = 0
        self._yaw = 0.0
        self._cmd_vx = 0.0
        self._cmd_vy = 0.0
        
        # ── AI Context (Mocking pacmanbot Ghost) ─────────────────────────────
        seed_val = int(os.environ.get('PACMAN_SEED', 42))
        random.seed(seed_val)
        np.random.seed(seed_val)
        self.grid, _ = generate_map()
        self._rows = len(self.grid)
        self._cols = len(self.grid[0])
        
        self.personal_map = np.full((self._rows, self._cols), UNKNOWN, dtype=np.int8)
        self.last_seen = np.full((self._rows, self._cols), -1, dtype=np.int64)
        self.known_agents = {i: "UNKNOWN" for i in range(MAX_GHOSTS)}
        self.frame = 0
        
        self.pacman_powered = False
        self.pacman_power_expiry_frame = 0
        self.known_pacman = None
        self.pacman_last_seen = -1
        self.last_lost_pacman = None
        self._pacman_pos_topic = None
        self.prev_pac_row = -1
        self.prev_pac_col = -1
        
        self._lidar_ranges = [4.0] * 360
        
        self.cbba_agent = CBBA_Agent(self.gid)
        self.belief_map = BeliefMap(self.gid, self.grid.copy())
        self.recent_nom = np.zeros((self._rows, self._cols), dtype=np.float32)
        
        self.message_queue = []
        self.last_heartbeat = {}
        self.seq = 0
        
        self.last_sync_frame = {}
        self._last_synced_map = {}
        
        # ── Navigation State ─────────────────────────────────────────────────
        self._cells_travelled = 0
        self._last_dir = (0, 1)
        self._nav_dir = (0, 1)
        self._target_row = 0
        self._target_col = 0
        self._arrived = False
        self._centering = False
        self._centering_axis = 'x'
        self._pending_target = (0, 0)
        self._seen_ids = set()
        self.dead = False
        
        from collections import deque
        self.pos_history = deque(maxlen=OSCILLATION_WINDOW)
        self._tail_pacman_remaining = 0

        # ── ROS Comm ─────────────────────────────────────────────────────────
        self._set_state = self.create_client(SetEntityState, '/set_entity_state')
        self._cmd_pub  = self.create_publisher(Twist, f'/{name}/cmd_vel', 10)
        self._nrf_pub  = self.create_publisher(String, f'/nrf24/{name}/tx', 20)
        self.create_subscription(Odometry, f'/{name}/odom',  self._odom_cb,  10)
        self.create_subscription(String,   f'/nrf24/{name}/rx', self._nrf_rx_cb, 20)
        self.create_subscription(Point,    '/pacman_bot/location',    self._pac_loc_cb, 10)
        self.create_subscription(Bool,     '/pacman_bot/power_state', self._pac_pow_cb, 10)
        
        self.create_subscription(LaserScan, f'/{name}/scan/front', lambda m, off=math.pi/2: self._scan_cb(m, off), 10)
        self.create_subscription(LaserScan, f'/{name}/scan/back', lambda m, off=-math.pi/2: self._scan_cb(m, off), 10)
        self.create_subscription(LaserScan, f'/{name}/scan/left', lambda m, off=math.pi: self._scan_cb(m, off), 10)
        self.create_subscription(LaserScan, f'/{name}/scan/right', lambda m, off=0.0: self._scan_cb(m, off), 10)
        self.create_subscription(String, '/game_events', self._game_events_cb, 10)

        # ── Timers ───────────────────────────────────────────────────────────
        self.create_timer(1.0 / 30.0, self._control_loop)
        
        # Initialize target to current pose when odom starts arriving
        self._initialized = False
        self.get_logger().info(f'Ghost node "{name}" AI ready')

    def _game_events_cb(self, msg: String):
        if msg.data == f"kill:{self.gid}":
            self.dead = True
            self.get_logger().info(f'Received kill event for ghost {self.gid}. Shutting down AI.')

    def _odom_cb(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._z = msg.pose.pose.position.z
        self._q = msg.pose.pose.orientation
        r, c = world_to_grid(self._x, self._y)
        
        q = self._q
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny_cosp, cosy_cosp)
        
        if not self._initialized:
            self.row = r
            self.col = c
            self._target_row = r
            self._target_col = c
            self._initialized = True

    def _nrf_rx_cb(self, msg: String):
        try:
            pkt = json.loads(msg.data)
        except Exception:
            return
        mid = pkt.get('id')
        if mid is None:
            return
        mid_tuple = tuple(mid)
        if mid_tuple in self._seen_ids:
            return
        self._seen_ids.add(mid_tuple)
        if len(self._seen_ids) > 500:
            for k in list(self._seen_ids)[:250]:
                self._seen_ids.discard(k)
        self.message_queue.append(pkt)

    def _pac_loc_cb(self, msg: Point):
        self._pacman_pos_topic = (int(round(msg.x)), int(round(msg.y)))

    def _pac_pow_cb(self, msg: Bool):
        self._pacman_pow_topic = msg.data

    @property
    def pacman_power_timer(self):
        if not getattr(self, 'pacman_powered', False):
            return 0
        rem_frames = self.pacman_power_expiry_frame - self.frame
        return max(0, int((rem_frames / 210.0) * 40.0))

    def _scan_cb(self, msg, yaw_offset):
        angle = msg.angle_min
        for r in msg.ranges:
            val = r
            if not math.isfinite(val) or val > msg.range_max:
                val = 4.0
            elif val < msg.range_min:
                val = msg.range_min
            
            global_angle = angle + yaw_offset
            idx = int(round(global_angle * 180.0 / math.pi)) % 360
            self._lidar_ranges[idx] = val
            angle += msg.angle_increment

    def _update_personal_map(self):
        visible = {}
        visible[(self.row, self.col)] = self.grid[self.row, self.col]
        
        pacman_spotted = False
        
        for i in range(360):
            r = self._lidar_ranges[i]
            if r > 4.0 or not math.isfinite(r):
                r = 4.0
                
            angle = math.radians(i) + self._yaw
            from maze_generator import world_to_grid, CELL_SIZE
            
            # Robust physical dynamic object detection
            if r < 4.0:
                hx = self._x + r * math.cos(angle)
                hy = self._y + r * math.sin(angle)
                hr, hc = world_to_grid(hx, hy)
                if 0 <= hr < self._rows and 0 <= hc < self._cols:
                    if self.grid[hr, hc] != WALL:
                        if self._pacman_pos_topic:
                            pr, pc = self._pacman_pos_topic
                            if abs(hr - pr) + abs(hc - pc) <= 1:
                                pacman_spotted = True

            # Map Discovery (Arcade Parity)
            # We explicitly ignore physical 'r' here to eliminate ghost points/shadows.
            # We always raycast up to MAX_RAY_DIST (4.0m) or until we hit a WALL,
            # ensuring ghosts are "transparent" to map mapping, exactly like pacmanbot.
            ray_dist = 4.0
            steps = max(1, int(ray_dist / (CELL_SIZE / 4.0)))
            
            for step in range(1, steps + 1):
                px = self._x + (ray_dist * step / steps) * math.cos(angle)
                py = self._y + (ray_dist * step / steps) * math.sin(angle)
                pr, pc = world_to_grid(px, py)
                if 0 <= pr < self._rows and 0 <= pc < self._cols:
                    visible[(pr, pc)] = self.grid[pr, pc]
                    if self.grid[pr, pc] == WALL:
                        break
                        
        if self._pacman_pos_topic and self._pacman_pos_topic in visible:
            pacman_spotted = True
                    
        diffs = []
        for (r, c), val in visible.items():
            if self.personal_map[r, c] != val:
                self.personal_map[r, c] = val
                if val == WALL:
                    self.belief_map.update_local_map_cell((r, c), WALL)
                diffs.append(("cell", int(r), int(c), int(val)))
            self.last_seen[r, c] = self.frame
            
        pacman_diff = None
        if pacman_spotted and self._pacman_pos_topic:
            pow_st = getattr(self, '_pacman_pow_topic', False)
            if self.known_pacman != self._pacman_pos_topic or self.pacman_powered != pow_st:
                self.known_pacman = self._pacman_pos_topic
                if pow_st and not self.pacman_powered:
                    self.pacman_power_expiry_frame = self.frame + 210
                self.pacman_powered = pow_st
                if not pow_st:
                    self.pacman_power_expiry_frame = 0
                self.pacman_last_seen = self.frame
                pacman_diff = ("pacman", int(self.known_pacman[0]), int(self.known_pacman[1]), bool(self.pacman_powered), int(self.frame))
            else:
                self.pacman_last_seen = self.frame
        elif self.known_pacman and self.known_pacman in visible:
            self.last_lost_pacman = self.known_pacman
            self.pacman_last_seen = self.frame
            pacman_diff = ("pacman_lost", int(self.known_pacman[0]), int(self.known_pacman[1]), int(self.frame))
            self.known_pacman = None
            
        if pacman_diff:
            diffs.append(pacman_diff)
            
        pr, pc = self._pacman_pos_topic if self._pacman_pos_topic else (-1, -1)
        pacman_in_los = pacman_spotted
        pacman_just_lost = pacman_diff is not None and pacman_diff[0] == "pacman_lost"
        
        if pacman_in_los:
            pac_dir = (0, 0)
            if self.prev_pac_row >= 0:
                pac_dir = (pr - self.prev_pac_row, pc - self.prev_pac_col)
            self.belief_map.observe((pr, pc), pac_dir)
            self.prev_pac_row, self.prev_pac_col = pr, pc
        elif pacman_just_lost:
            _, kr, kc, _ = pacman_diff
            self.belief_map.observe_lost((kr, kc))
            
        pac_pos = (pr, pc) if pacman_in_los else None
        self.belief_map.observe_clear(set(visible.keys()), pac_pos)
            
        return diffs

    def _check_liveness(self):
        for gid in list(self.last_heartbeat.keys()):
            if self.frame - self.last_heartbeat[gid] > HEARTBEAT_TIMEOUT:
                if self.known_agents.get(gid) != "UNKNOWN":
                    self.known_agents[gid] = "UNKNOWN"
                    self._broadcast_nrf([("agent_lost", gid)])

    def _send_full_sync(self, target_gid):
        last = self._last_synced_map.get(target_gid)
        if last is not None:
            changed = (self.personal_map != last) & (self.personal_map != UNKNOWN)
            rs, cs = np.nonzero(changed)
        else:
            mask = self.personal_map != UNKNOWN
            rs, cs = np.nonzero(mask)
            
        sync_diffs = [("cell", int(r), int(c), int(self.personal_map[r, c])) for r, c in zip(rs, cs)]
        self._last_synced_map[target_gid] = self.personal_map.copy()
        
        for gid, pos in self.known_agents.items():
            if pos == "UNKNOWN":
                sync_diffs.append(("agent_lost", gid))
            elif pos is not None:
                sync_diffs.append(("agent", gid, int(pos[0]), int(pos[1])))
                
        for gid, hb_frame in self.last_heartbeat.items():
            frames_ago = self.frame - hb_frame
            sync_diffs.append(("hb_sync", gid, frames_ago))
            
        if self.known_pacman is not None:
            sync_diffs.append(("pacman", int(self.known_pacman[0]), int(self.known_pacman[1]), bool(self.pacman_powered), int(self.pacman_last_seen)))
        elif self.last_lost_pacman is not None and self.pacman_last_seen > -1:
            sync_diffs.append(("pacman_lost", int(self.last_lost_pacman[0]), int(self.last_lost_pacman[1]), int(self.pacman_last_seen)))
            
        if sync_diffs:
            sync_id = ["sync", self.gid, target_gid, int(self.frame)]
            self._broadcast_nrf(sync_diffs, msg_id=sync_id, hop=0)

    def _process_nrf_messages(self):
        for pkt in self.message_queue:
            mid = pkt.get('id')
            if not mid: continue
            
            if mid[0] == "sync" and mid[2] != self.gid:
                continue

            sender = mid[1] if mid[0] == "sync" else mid[0]
            if sender != self.gid:
                last_sync = self.last_sync_frame.get(sender, -1)
                if self.frame - last_sync >= RESYNC_EVERY:
                    self.last_sync_frame[sender] = self.frame
                    self._send_full_sync(sender)

            hop = pkt.get('hop', 0)
            relay_diffs = []
            
            for diff in pkt.get('diffs', []):
                dtype = diff[0]
                if dtype == "cell":
                    _, r, c, val = diff
                    old = self.personal_map[r, c]
                    if old != val:
                        if old != UNKNOWN and self.last_seen[r, c] >= self.frame - MEMORY_FRAMES:
                            continue
                        self.personal_map[r, c] = val
                        if val == PELLET and self.grid[r, c] == POWER:
                            self.grid[r, c] = PELLET
                        if val == WALL:
                            self.belief_map.update_local_map_cell((r, c), WALL)
                        relay_diffs.append(diff)
                elif dtype == "agent":
                    _, gid, r, c = diff
                    if gid == self.gid: continue
                    old = self.known_agents.get(gid)
                    if old != (r, c):
                        self.known_agents[gid] = (r, c)
                        relay_diffs.append(diff)
                elif dtype == "agent_lost":
                    _, gid = diff
                    if gid == self.gid: continue
                    if self.known_agents.get(gid) != "UNKNOWN":
                        self.known_agents[gid] = "UNKNOWN"
                        relay_diffs.append(diff)
                elif dtype == "heartbeat":
                    _, gid, r, c, origin_frame = diff
                    if gid == self.gid: continue
                    existing = self.last_heartbeat.get(gid, -1)
                    if origin_frame > existing:
                        self.last_heartbeat[gid] = origin_frame
                    if r != 0 or c != 0:
                        old = self.known_agents.get(gid)
                        if old != (r, c):
                            self.known_agents[gid] = (r, c)
                            relay_diffs.append(("agent", gid, r, c))
                    relay_diffs.append(diff)
                elif dtype == "hb_sync":
                    _, gid, frames_ago = diff
                    if gid == self.gid: continue
                    reconstructed = self.frame - frames_ago
                    existing = self.last_heartbeat.get(gid, -1)
                    if reconstructed > existing:
                        self.last_heartbeat[gid] = reconstructed
                        relay_diffs.append(diff)
                elif dtype == "pacman":
                    _, pr, pc, pow_st, pf = diff
                    if pf > self.pacman_last_seen:
                        self.known_pacman = (pr, pc)
                        if pow_st and not self.pacman_powered:
                            self.pacman_power_expiry_frame = self.frame + 210
                        self.pacman_powered = pow_st
                        if not pow_st:
                            self.pacman_power_expiry_frame = 0
                        self.pacman_last_seen = pf
                        self.last_lost_pacman = None
                        relay_diffs.append(diff)
                elif dtype == "pacman_lost":
                    _, pr, pc, pf = diff
                    if pf > self.pacman_last_seen:
                        if self.known_pacman == (pr, pc):
                            self.known_pacman = None
                        self.last_lost_pacman = (pr, pc)
                        self.pacman_last_seen = pf
                        relay_diffs.append(diff)
                elif dtype == "cbba":
                    _, gid, payload = diff
                    if gid == self.gid: continue
                    changed = self.cbba_agent.receive_consensus(gid, payload.get("y", {}), payload.get("z", {}), payload.get("s", {}), self.frame)
                    if changed:
                        relay_diffs.append(diff)
                elif dtype == "belief":
                    _, gid, payload = diff
                    if gid == self.gid: continue
                    self.belief_map.merge(gid, payload, self.frame)
                    relay_diffs.append(diff)
                    
            if relay_diffs:
                MAX_RELAY_SIZE = 50
                for idx, i in enumerate(range(0, len(relay_diffs), MAX_RELAY_SIZE)):
                    chunk = relay_diffs[i: i + MAX_RELAY_SIZE]
                    if idx == 0:
                        chunk_msg_id = mid
                    else:
                        chunk_msg_id = list(mid) + [f"chunk_{idx}"]
                    self._broadcast_nrf(chunk, msg_id=chunk_msg_id, hop=hop+1)
                    
        self.message_queue.clear()
        self.belief_map._ensure_initialised()

    def _broadcast_nrf(self, diffs, msg_id=None, hop=0):
        if not diffs and msg_id is not None:
            return
            
        is_new_msg = (msg_id is None)
        if is_new_msg:
            msg_id = [self.gid, int(self.frame), self.seq]
            self.seq += 1
            cbba_payload = self.cbba_agent.get_consensus_payload()
            belief_payload = self.belief_map.get_payload()
            diffs = list(diffs) + [("cbba", self.gid, cbba_payload), ("belief", self.gid, belief_payload)]
            
        mid_tuple = tuple(msg_id)
        self._seen_ids.add(mid_tuple)
        
        pkt = {
            'id': msg_id,
            'diffs': diffs,
            'hop': hop
        }
        
        self._nrf_pub.publish(String(data=json.dumps(pkt)))

    def _choose_next_dir(self):
        self._cells_travelled += 1
        if self._cells_travelled % 3 == 0:
            sp = build_spatial(self, self.recent_nom, self._rows, self._cols)
            ve = build_vector(self)
            vm = build_valid_mask(self, self._rows, self._cols)
            with torch.inference_mode():
                idx, _, scores, _, _ = RL_ACTOR(
                    torch.tensor(sp, device=RL_DEVICE).unsqueeze(0),
                    torch.tensor(ve, device=RL_DEVICE).unsqueeze(0),
                    torch.tensor(vm, device=RL_DEVICE).unsqueeze(0),
                    K=3
                )
            idx_np = idx.cpu().numpy()[0]
            scores_map = scores.cpu().numpy()[0]
            indices = [(int(x // self._cols), int(x % self._cols)) for x in idx_np]
            self.recent_nom *= 0.8
            for r, c in indices:
                if 0 <= r < self._rows and 0 <= c < self._cols:
                    self.recent_nom[r, c] = 1.0
            
            tasks = actions_to_tasks(self, scores_map, indices, self.frame)
            self.cbba_agent._last_auction = self.frame + AUCTION_EVERY
            
            all_tasks, h_dists = generate_tasks(self, self.frame)
            all_tasks.extend(tasks)
            all_tasks.sort(key=lambda t: t.score, reverse=True)
            self.cbba_agent._task_map.clear()
            if tasks:
                all_targets = [t.target_pos for t in tasks]
                dists = pathfinder.dijkstra_multi(self.grid, (self.row, self.col), all_targets)
                h_dists.update(dists)
            self.cbba_agent._phase1(self, all_tasks, h_dists)

        active_task = self.cbba_agent.step(self, self.frame)
        if active_task and self.pacman_powered and active_task.task_type == TaskType.HUNT:
            active_task = None
            
        if active_task is not None and (self.row, self.col) == active_task.target_pos:
            key = (int(active_task.task_type), active_task.target_pos, getattr(active_task, 'owner', -1))
            if key in self.cbba_agent.path: self.cbba_agent.path.remove(key)
            if key in self.cbba_agent.bundle: self.cbba_agent.bundle.remove(key)
            active_task = None

        best_dir = None
        if not self.pacman_powered and self.known_pacman:
            pr, pc = self.known_pacman
            if abs(self.row - pr) + abs(self.col - pc) == 1:
                best_dir = (pr - self.row, pc - self.col)
                self.cbba_agent.bundle.clear()
                self.cbba_agent.path.clear()
                if hasattr(self, '_committed_path'):
                    self._committed_path = []
                
                nearby_ghosts = 1
                for _gid, pos in self.known_agents.items():
                    if pos != "UNKNOWN":
                        if abs(pos[0] - self.row) + abs(pos[1] - self.col) <= 6:
                            nearby_ghosts += 1
                self._tail_pacman_remaining = 2 if nearby_ghosts <= 2 else 0

        if not best_dir and self._tail_pacman_remaining > 0:
            if self.known_pacman and not self.pacman_powered:
                pr, pc = self.known_pacman
                best_dist = abs(self.row - pr) + abs(self.col - pc)
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = self.row + dr, self.col + dc
                    if 0 <= nr < self._rows and 0 <= nc < self._cols and self.grid[nr][nc] != WALL:
                        d = abs(nr - pr) + abs(nc - pc)
                        if d < best_dist:
                            best_dist = d
                            best_dir = (dr, dc)
                if best_dir is not None:
                    self._tail_pacman_remaining -= 1
                    if hasattr(self, '_committed_path'):
                        self._committed_path = []
                else:
                    self._tail_pacman_remaining = 0
            else:
                self._tail_pacman_remaining = 0
                
        if not best_dir:
            dist_to_pac = abs(self.row - self.known_pacman[0]) + abs(self.col - self.known_pacman[1]) if self.known_pacman else 999
            if dist_to_pac > 3 or self.pacman_powered:
                from collections import deque
                queue = deque([(self.row, self.col, 0, [])])
                visited = {(self.row, self.col)}
                while queue:
                    r, c, d, path = queue.popleft()
                    if self.grid[r][c] == POWER and d > 0:
                        if d == 1 or (d == 2 and (dist_to_pac > 6 or self.pacman_powered)):
                            best_dir = path[0]
                            if hasattr(self, '_committed_path'):
                                self._committed_path = []
                            break
                    if d < 2:
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self._rows and 0 <= nc < self._cols and self.grid[nr][nc] != WALL and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                queue.append((nr, nc, d + 1, path + [(dr, dc)]))
                
        if not best_dir and active_task:
            target = active_task.target_pos
            if getattr(self, '_committed_target', None) != target or not getattr(self, '_committed_path', []):
                path = pathfinder.astar(self.grid, (self.row, self.col), target)
                if path and len(path) >= 2:
                    self._committed_path = path[1:]
                    self._committed_target = target
                else:
                    self._committed_path = []
            
            nxt = None
            while hasattr(self, '_committed_path') and self._committed_path:
                cand = self._committed_path.pop(0)
                if self.grid[cand[0], cand[1]] != WALL:
                    nxt = cand
                    break
                else:
                    path = pathfinder.astar(self.grid, (self.row, self.col), target)
                    if path and len(path) >= 2:
                        self._committed_path = path[1:]
                        self._committed_target = target
                        nxt = self._committed_path.pop(0)
                    else:
                        self._committed_path = []
                    break
            
            if nxt is not None and nxt != (self.row, self.col) and self.grid[nxt[0], nxt[1]] != WALL:
                if self.pacman_powered and self.known_pacman is not None and nxt == self.known_pacman:
                    pass
                else:
                    best_dir = (nxt[0] - self.row, nxt[1] - self.col)
                
        if not best_dir:
            valid = []
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = self.row + dr, self.col + dc
                if 0 <= nr < self._rows and 0 <= nc < self._cols and self.grid[nr][nc] != WALL:
                    if (dr, dc) != (-self._last_dir[0], -self._last_dir[1]):
                        valid.append((dr, dc))
            if valid:
                import random
                best_dir = random.choice(valid)
            else:
                best_dir = (-self._last_dir[0], -self._last_dir[1])
                
        self._last_dir = best_dir
        
        self.pos_history.append((self.row, self.col))
        self._check_oscillation()
        
        return best_dir

    def _check_oscillation(self):
        if len(self.pos_history) < OSCILLATION_WINDOW:
            return
        cur = (self.row, self.col)
        if self.pos_history.count(cur) >= 2:
            if self.known_pacman is None and self.last_lost_pacman is not None:
                self.last_lost_pacman = None
                self.pos_history.clear()
        if self.pos_history.count(cur) >= 3:
            self.cbba_agent.bundle.clear()
            self.cbba_agent.path.clear()
            self.pos_history.clear()
            if hasattr(self, '_committed_path'):
                self._committed_path = []

    def _compute_cmd(self, tx: float, ty: float) -> Twist:
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
            p_gain = BOT_SPEED / ARRIVE_DIST
            if self._centering_axis == 'x':
                vx_raw = err_x * p_gain
                vx = math.copysign(max(DECEL_MIN_SPD, min(BOT_SPEED, abs(vx_raw))), vx_raw)
                vy = err_y * CROSS_KP
            else:
                vy_raw = err_y * p_gain
                vx = err_x * CROSS_KP
                vy = math.copysign(max(DECEL_MIN_SPD, min(BOT_SPEED, abs(vy_raw))), vy_raw)
        else:
            speed = BOT_SPEED
            
            if dc != 0:
                vx = math.copysign(speed, err_x)
                vy = err_y * CROSS_KP
            elif dr != 0:
                vy = math.copysign(speed, err_y)
                vx = err_x * CROSS_KP

        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        cmd.linear.x =  vx * cos_y + vy * sin_y
        cmd.linear.y = -vx * sin_y + vy * cos_y

        cmd.angular.z = -15.0 * self._yaw

        return cmd

    def _teleport_self(self, x, y, force_z=None):
        self._x = x
        self._y = y
        if self._set_state.service_is_ready():
            req = SetEntityState.Request()
            req.state.name = self._name
            req.state.pose.position.x = float(x)
            req.state.pose.position.y = float(y)
            req.state.pose.position.z = force_z if force_z is not None else float(self._z)
            if self._q is not None and force_z is None:
                req.state.pose.orientation = self._q
            else:
                req.state.pose.orientation.w = 1.0
            self._set_state.call_async(req)

    def _control_loop(self):
        if not self._initialized: return
        
        if getattr(self, 'dead', False):
            self._cmd_pub.publish(Twist())
            return
        
        # Kinetic/tick based absolute frame syncing to Gazebo clock
        current_time_ns = self.get_clock().now().nanoseconds
        self.frame = int(current_time_ns / (1e9 / 30.0))
        
        if self.pacman_powered and self.frame >= getattr(self, 'pacman_power_expiry_frame', 999999):
            self.pacman_powered = False

        if self.frame % 3 == 0:
            self._check_liveness()
            diffs = self._update_personal_map()
            if self.frame % HEARTBEAT_EVERY == 0:
                diffs.append(("heartbeat", self.gid, int(self.row), int(self.col), int(self.frame)))
            self._broadcast_nrf(diffs)
            self._process_nrf_messages()
            self.belief_map.update_safety_map(self.known_agents, self.frame, self.pacman_powered)
            self.belief_map.diffuse((self.row, self.col))

        tx, ty = cell_center_world(self._target_row, self._target_col)
        dist   = math.hypot(self._x - tx, self._y - ty)

        if dist < ARRIVE_DIST:
            if not self._arrived:
                self._arrived = True
                old_dir = self._nav_dir
                intersect_r = self._target_row
                intersect_c = self._target_col
                self.row = intersect_r
                self.col = intersect_c
                
                if self.grid[self.row, self.col] == POWER:
                    self.grid[self.row, self.col] = PELLET
                
                
                dr, dc = self._choose_next_dir()
                self._nav_dir = (dr, dc)
                
                next_r = self._target_row + dr
                next_c = self._target_col + dc

                if self._nav_dir != old_dir:
                    self._pending_target = (next_r, next_c)
                    self._target_row = intersect_r
                    self._target_col = intersect_c
                    self._centering  = True
                    self._centering_axis = 'x' if old_dir[1] != 0 else 'y'
                else:
                    self._target_row = next_r
                    self._target_col = next_c
                tx, ty = cell_center_world(self._target_row, self._target_col)
        else:
            self._arrived = False

        if self._centering:
            cx, cy = cell_center_world(self._target_row, self._target_col)
            err_c = (self._x - cx) if self._centering_axis == 'x' else (self._y - cy)
            if abs(err_c) < CELL_SNAP_DIST:
                self._centering  = False
                # removed teleport snapping for fluid physical motion
                self._target_row = self._pending_target[0]
                self._target_col = self._pending_target[1]
                self._arrived    = False
                tx, ty = cell_center_world(self._target_row, self._target_col)

        self._cmd_pub.publish(self._compute_cmd(tx, ty))

    def kill(self):
        self._dead = True
        self._dead_expiry_frame = self.frame + 60

    def _stop(self):
        if rclpy.ok():
            self._cmd_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
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
