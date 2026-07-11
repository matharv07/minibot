#!/usr/bin/env python3
"""pacman_node.py — drives /pacman with Player AI, cell-by-cell navigation."""
import sys, os, json, math, types
import numpy as np

# ── Mock pygame / heavy deps ───────────────────────────────────────────────
_fp = types.ModuleType('pygame')
_fp.Rect = lambda *a,**k: None; _fp.init = lambda: None; _fp.SRCALPHA = 0
_fp.QUIT = _fp.KEYDOWN = _fp.MOUSEBUTTONDOWN = 0
for _k in ('K_w','K_a','K_s','K_d','K_UP','K_DOWN','K_LEFT','K_RIGHT',
           'K_r','K_0','K_1','K_2','K_3','K_4','K_5','K_6'):
    setattr(_fp, _k, 0)
_fp.font = types.SimpleNamespace(init=lambda:None, SysFont=lambda*a,**k:None,
                                  Font=lambda*a,**k:None)
_fp.display = types.SimpleNamespace(set_mode=lambda*a,**k:None,
                                     set_caption=lambda*a,**k:None)
_fp.draw = types.SimpleNamespace(); _fp.Surface = lambda*a,**k: None
sys.modules.update({'pygame':_fp,'pygame.font':_fp.font,
                    'pygame.display':_fp.display,'pygame.draw':_fp.draw})

for _m in ('torch','ghost','pathfinder','cbba','beliefmap',
           'allocator','obs','net','curriculum','setup_dependencies'):
    if _m not in sys.modules:
        _s = types.ModuleType(_m)
        if _m == 'ghost':
            _s.Ghost = type('Ghost',(),{'__init__':lambda*a,**k:None}); _s.UNKNOWN=-1
        elif _m == 'pathfinder':
            _s.build_scipy_graph=lambda*a,**k:None
            _s._SCIPY_AVAILABLE=False; _s.get_scipy_graph=lambda*a,**k:None
        elif _m == 'curriculum':
            from dataclasses import dataclass
            @dataclass(frozen=True)
            class _St:
                rows:int; cols:int; n_ghosts:int; n_power:int
                advance_return:float; min_updates:int
            _s.STAGES=[_St(7,9,2,2,42.,150),_St(13,17,3,6,40.,350),
                       _St(21,27,5,14,28.,360),_St(27,33,6,24,22.,400),
                       _St(33,41,7,28,float('inf'),0)]
        elif _m == 'net': _s.GhostActor=type('GhostActor',(),{})
        elif _m == 'setup_dependencies': _s.main=lambda:None
        sys.modules[_m]=_s

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, Bool, String, Int32MultiArray
from ament_index_python.packages import get_package_share_directory

def _add_path():
    try:
        pb = os.path.join(get_package_share_directory('minibot'),'pacmanbot')
    except Exception:
        pb = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            '..','pacmanbot'))
    if os.path.isdir(pb) and pb not in sys.path:
        sys.path.insert(0, pb)
_add_path()

import pacman as _pac
from pacman import Player, WALL, EMPTY, PELLET, POWER
_pac.AUTO_MODE = True

ARRIVAL = 0.22      # metres — snap threshold to count "arrived at cell"
N_GHOSTS = 7


class PacmanGameNode(Node):
    def __init__(self):
        super().__init__('pacman_game_node')
        self.declare_parameter('grid_json',    '')
        self.declare_parameter('player_start', '16,20')
        self.declare_parameter('cell_pitch',   1.09)
        self.declare_parameter('speed',        1.5)

        grid_json  = self.get_parameter('grid_json').value
        start_str  = self.get_parameter('player_start').value
        self.cp    = self.get_parameter('cell_pitch').value
        self.speed = self.get_parameter('speed').value

        if not grid_json:
            raise RuntimeError('grid_json param missing')

        grid_list = json.loads(grid_json)
        self._np  = np.array(grid_list, dtype=np.int8)   # ground-truth grid
        grid_ll   = [list(r) for r in grid_list]
        self._R, self._C = len(grid_ll), len(grid_ll[0])

        pr, pc = map(int, start_str.split(','))
        self.player = Player(grid_ll, (pr, pc))

        # Odom state — initialised to spawn position
        self.pac_x = self._wx(pc); self.pac_y = self._wy(pr)
        self.pac_yaw = 0.0
        self._got_odom = False

        # Committed target cell (cell-by-cell navigation)
        self._tr = pr; self._tc = pc          # current target grid cell
        self._tx = self._wx(pc); self._ty = self._wy(pr)  # target world pos

        # Total pellet counts
        self.total_pellets = int(np.sum(self._np == PELLET))
        self.total_power   = int(np.sum(self._np == POWER))
        self._dir_r = 0; self._dir_c = 0
        self._arrived   = True   # True = ready to pick next cell
        self._pellet_ok = False  # True = halfway pellet already collected for this cell
        self._steps  = 0
        self._state  = 'playing'

        # Ghost cell positions
        self.ghost_cells = {}

        # Lazy delete-entity client
        self._del_cli = None

        # Publishers
        self.pac_cmd  = self.create_publisher(Twist,  '/pacman/cmd_vel', 10)
        self.score_p  = self.create_publisher(Int32,  '/game/score',     10)
        self.state_p  = self.create_publisher(String, '/game/state',     10)
        self.power_p  = self.create_publisher(Bool,   '/game/powered',   10)
        self.steps_p  = self.create_publisher(Int32,  '/game/steps',     10)
        self.pellet_p = self.create_publisher(Int32MultiArray, '/game/pellets_remaining', 10)

        # Subscriptions
        self.create_subscription(Odometry,'/pacman/odom', self._pac_odom, 10)
        for i in range(N_GHOSTS):
            self.create_subscription(Odometry, f'/ghost_{i}/odom',
                lambda msg, i=i: self._ghost_odom(msg, i), 10)

        self.create_timer(0.05, self._loop)   # 20 Hz control loop
        self.get_logger().info(
            f'pacman_node ready  start=({pr},{pc})  cp={self.cp:.3f}m  speed={self.speed}m/s')

    # ── coordinate helpers ─────────────────────────────────────────────────
    def _wx(self, c): return c * self.cp + self.cp / 2.0
    def _wy(self, r): return -(r * self.cp + self.cp / 2.0)
    def _cell(self, x, y):
        c = int(round((x - self.cp/2) / self.cp))
        r = int(round((-y - self.cp/2) / self.cp))
        return max(0,min(r,self._R-1)), max(0,min(c,self._C-1))

    # ── odom callbacks ─────────────────────────────────────────────────────
    def _pac_odom(self, msg):
        self.pac_x = msg.pose.pose.position.x
        self.pac_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.pac_yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self._got_odom = True

    def _ghost_odom(self, msg, i):
        self.ghost_cells[f'ghost_{i}'] = self._cell(
            msg.pose.pose.position.x, msg.pose.pose.position.y)

    # ── pellet collection ──────────────────────────────────────────────────
    def _collect(self, r, c, who='pacman'):
        """Consume pellet at (r,c) using _np as ground truth."""
        val = int(self._np[r, c])
        if val not in (PELLET, POWER):
            return
        self._np[r, c] = EMPTY
        self.player.grid[r][c] = EMPTY          # keep Player grid in sync
        prefix = 'pel' if val == PELLET else 'pow'
        self._delete(f'{prefix}_{r}_{c}')
        if who == 'pacman':
            if val == POWER:
                self.player.powered    = True
                self.player.power_timer = 200 # 10s at 20Hz
                self.get_logger().info('⚡ POWER PELLET!')

    def _delete(self, name):
        try:
            from gazebo_msgs.srv import DeleteEntity
        except ImportError:
            return
        if self._del_cli is None:
            self._del_cli = self.create_client(DeleteEntity, '/delete_entity')
        if not self._del_cli.service_is_ready():
            return
        req = DeleteEntity.Request(); req.name = name
        self._del_cli.call_async(req)

    # ── main loop (20 Hz) ──────────────────────────────────────────────────
    def _loop(self):
        if self._state != 'playing' or not self._got_odom:
            return

        dist = math.hypot(self._tx - self.pac_x, self._ty - self.pac_y)

        # 1. Collect pellet when precisely crossing it (0.25m from center)
        if dist < 0.25 and not self._pellet_ok:
            self._pellet_ok = True
            self._collect(self._tr, self._tc)

        # 2. Arrival: reached cell centre (tight threshold)
        if dist < ARRIVAL and not self._arrived:
            self._arrived = True

        # 3. Pick next cell from Player AI (exactly once per arrival)
        if self._arrived:
            self.player.row = self._tr
            self.player.col = self._tc
            self.player.update({})
            new_r, new_c = self.player.row, self.player.col

            self._dir_r = new_r - self._tr
            self._dir_c = new_c - self._tc
            self._tr = new_r; self._tc = new_c
            self._tx = self._wx(new_c); self._ty = self._wy(new_r)
            self._arrived   = False
            self._pellet_ok = False
            self._steps += 1
            steps_msg = Int32(); steps_msg.data = self._steps
            self.steps_p.publish(steps_msg)

        # 4. Correct for physical rotation and drift
        dx = self._tx - self.pac_x
        dy = self._ty - self.pac_y
        dist2 = math.hypot(dx, dy)
        twist = Twist()

        # Fix yaw to 0 (P-controller)
        yaw_err = 0.0 - getattr(self, 'pac_yaw', 0.0)
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
        twist.angular.z = yaw_err * 3.0  # gentle rotation correction

        # Transform world error (dx, dy) to robot's local frame
        # If robot is rotated, we must apply forces in its rotated axes
        yaw = getattr(self, 'pac_yaw', 0.0)
        s = math.sin(-yaw)
        c = math.cos(-yaw)
        local_dx = dx * c - dy * s
        local_dy = dx * s + dy * c

        if dist2 > 0.04:
            # We determine dominant axis in the local frame
            if abs(local_dx) >= abs(local_dy):
                twist.linear.x = math.copysign(self.speed, local_dx)
                twist.linear.y = local_dy * 3.0  # gentle proportional correction on off-axis
            else:
                twist.linear.x = local_dx * 3.0  # gentle proportional correction on off-axis
                twist.linear.y = math.copysign(self.speed, local_dy)
        self.pac_cmd.publish(twist)

        # Real-time power timer countdown
        if self.player.powered:
            self.player.power_timer -= 1
            if self.player.power_timer <= 0:
                self.player.powered = False
        now_powered = bool(self.player.powered)

        # 6. Game state topics
        rem_pellets = int(np.sum(self._np == PELLET))
        rem_power   = int(np.sum(self._np == POWER))
        consumed_pellets = self.total_pellets - rem_pellets
        consumed_power   = self.total_power - rem_power

        if rem_pellets + rem_power == 0:
            self._state = 'win'
            self.get_logger().info(
                f'WIN! score={self.player.score} steps={self._steps}')
        score_msg = Int32(); score_msg.data = self.player.score
        self.score_p.publish(score_msg)
        st_msg = String();   st_msg.data = self._state
        self.state_p.publish(st_msg)
        pw_msg = Bool();     pw_msg.data = now_powered
        self.power_p.publish(pw_msg)
        
        pl_msg = Int32MultiArray()
        pl_msg.data = [consumed_pellets, self.total_pellets, consumed_power, self.total_power]
        self.pellet_p.publish(pl_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PacmanGameNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()

