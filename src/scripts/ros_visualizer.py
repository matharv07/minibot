#!/usr/bin/env python3
import os
import sys
import math
import random
import json
import threading
import ast

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from maze_generator import (
    world_to_grid, GHOST_NAMES, N_GHOSTS, 
    generate_map, compute_ghost_starts,
    CELL_SIZE
)

_PACMAN_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', 'pacmanbot'))
if _PACMAN_DIR not in sys.path:
    sys.path.insert(0, _PACMAN_DIR)

import pacman as pacman_module
from pacman import (
    ROWS, COLS, CELL, WIDTH, HEIGHT, FPS,
    BLACK, WHITE, YELLOW, BLUE, RED, PINK, CYAN, ORANGE, DKBLUE, GREY, POWERED_COLOR, GHOST_COLORS,
    WALL, EMPTY, PELLET, POWER
)
from ghost import UNKNOWN

class GhostState:
    def __init__(self):
        self.personal_map = np.full((ROWS, COLS), UNKNOWN, dtype=np.int8)
        self.belief_cells = {}
        self.known_agents = {i: "UNKNOWN" for i in range(N_GHOSTS)}
        self.known_pacman = None
        self.pacman_powered = False
        self.row = 0
        self.col = 0

class VisualizerNode(Node):
    def __init__(self, game):
        super().__init__('ros_visualizer')
        self.game = game
        
        self.create_subscription(String, '/pacman_bot/stats', self.stats_cb, 10)
        self.create_subscription(Odometry, '/pacman_bot/odom', self.pacman_odom_cb, 10)
        self.create_subscription(String, '/game_events', self.game_events_cb, 10)
        
        self.ghost_odom_subs = {}
        self.ghost_tx_subs = {}
        for i, gname in enumerate(GHOST_NAMES):
            self.ghost_odom_subs[gname] = self.create_subscription(
                Odometry, f'/{gname}/odom', 
                lambda msg, gid=i: self.ghost_odom_cb(msg, gid), 
                10
            )
            self.ghost_tx_subs[gname] = self.create_subscription(
                String, f'/nrf24/{gname}/tx',
                lambda msg, gid=i: self.ghost_tx_cb(msg, gid),
                20
            )

    def stats_cb(self, msg):
        try:
            stats = json.loads(msg.data)
            self.game.score = stats.get('score', 0)
            self.game.powered = stats.get('powered', False)
            self.game.power_timer = stats.get('power_timer', 0)
            
            pr, pc = stats.get('row', -1), stats.get('col', -1)
            if 0 <= pr < ROWS and 0 <= pc < COLS:
                if self.game.grid[pr][pc] in (PELLET, POWER):
                    self.game.grid[pr][pc] = EMPTY
        except Exception:
            pass

    def game_events_cb(self, msg):
        try:
            if msg.data.startswith('kill:'):
                gid = int(msg.data.split(':')[1])
                self.game.ghost_positions[gid] = None
                self.game.dead_ghosts.add(gid)
            elif msg.data.startswith('eat:'):
                parts = msg.data.split(':')
                r, c = int(parts[1]), int(parts[2])
                if 0 <= r < ROWS and 0 <= c < COLS:
                    self.game.grid[r][c] = EMPTY
        except Exception:
            pass

    def pacman_odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.game.pacman_pos = (x, y)
        self.game.pacman_yaw = yaw

    def ghost_odom_cb(self, msg, gid):
        if gid in self.game.dead_ghosts: return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.game.ghost_positions[gid] = (x, y)
        r, c = world_to_grid(x, y)
        self.game.ghost_states[gid].row = r
        self.game.ghost_states[gid].col = c

    def ghost_tx_cb(self, msg, gid):
        try:
            pkt = json.loads(msg.data)
            diffs = pkt.get('diffs', [])
            gs = self.game.ghost_states[gid]
            for diff in diffs:
                dtype = diff[0]
                if dtype == "cell":
                    _, r, c, val = diff
                    gs.personal_map[r, c] = val
                elif dtype == "agent":
                    _, gid2, r, c = diff
                    gs.known_agents[gid2] = (r, c)
                elif dtype == "agent_lost":
                    _, gid2 = diff
                    gs.known_agents[gid2] = "UNKNOWN"
                elif dtype == "pacman":
                    _, pr, pc, pow_st, pf = diff
                    gs.known_pacman = (pr, pc)
                    gs.pacman_powered = pow_st
                elif dtype == "pacman_lost":
                    gs.known_pacman = None
                elif dtype == "belief":
                    _, gid2, payload = diff
                    if gid2 == gid: # ghost's own belief map
                        cells = payload.get("cells", {})
                        new_belief = {}
                        for k, v in cells.items():
                            if isinstance(k, str):
                                try:
                                    k_tuple = ast.literal_eval(k)
                                    new_belief[k_tuple] = v
                                except Exception:
                                    pass
                            else:
                                new_belief[k] = v
                        gs.belief_cells = new_belief
        except Exception:
            pass


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH * 2, HEIGHT))
        pygame.display.set_caption("PACMAN (ROS 2 Visualizer)")
        self.clock = pygame.time.Clock()
        try:
            pygame.font.init()
            self.font  = pygame.font.SysFont("monospace", 18, bold=True)
            self.small = pygame.font.SysFont("monospace", 14)
        except Exception:
            self.font  = pygame.font.Font(None, 22)
            self.small = pygame.font.Font(None, 16)
        
        self.score = 0
        self.powered = False
        self.power_timer = 0
        self.pacman_pos = None
        self.pacman_yaw = 0.0
        self.ghost_positions = {i: None for i in range(N_GHOSTS)}
        self.ghost_states = {i: GhostState() for i in range(N_GHOSTS)}
        self.dead_ghosts = set()
        self.mouth_tick = 0
        self.mouth_open = True
        self.debug_ghost_id = 0
        
        import glob
        ckpts = glob.glob(os.path.join(_PACMAN_DIR, "checkpoints/ckpt_*.pt"))
        self.rl_mode = len(ckpts) > 0
        self.auto_mode = True  # pacman_node operates autonomously
        
        self.new_game()

    def new_game(self):
        seed_val = int(os.environ.get('PACMAN_SEED', 42))
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        self.grid, self.player_start = generate_map(seed=seed_val)
        self.ghost_starts = compute_ghost_starts(self.grid, self.player_start, N_GHOSTS)
        self.dead_ghosts.clear()

    def _world_to_screen(self, wx, wy):
        col = wx / CELL_SIZE + COLS / 2.0 - 0.5
        row = ROWS / 2.0 - wy / CELL_SIZE - 0.5
        return col * CELL, row * CELL

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                    num = event.key - pygame.K_0
                    if num in self.ghost_states:
                        self.debug_ghost_id = num

    def update(self):
        self.mouth_tick += 1
        if self.mouth_tick >= 3:
            self.mouth_tick = 0
            self.mouth_open = not self.mouth_open

    def draw_grid(self):
        surf = self.screen
        for r in range(ROWS):
            for c in range(COLS):
                x = c * CELL
                y = r * CELL
                cell = self.grid[r][c]
                if cell == WALL:
                    pygame.draw.rect(surf, DKBLUE, (x, y, CELL, CELL))
                    pygame.draw.rect(surf, BLUE, (x + 1, y + 1, CELL - 2, CELL - 2))
                else:
                    pygame.draw.rect(surf, BLACK, (x, y, CELL, CELL))
                    if cell == PELLET:
                        pygame.draw.circle(surf, WHITE, (x + CELL // 2, y + CELL // 2), 2)
                    elif cell == POWER:
                        pygame.draw.circle(surf, WHITE, (x + CELL // 2, y + CELL // 2), 5)

    def draw_hud(self):
        y = ROWS * CELL
        pygame.draw.rect(self.screen, BLACK, (0, y, WIDTH * 2, 48))
        score_txt = self.font.render(f"SCORE  {self.score}", True, WHITE)
        self.screen.blit(score_txt, (10, y + 6))
        if self.powered:
            bar_w = int((self.power_timer / 40) * 100)
            pygame.draw.rect(self.screen, GREY, (WIDTH // 2 - 50, y + 28, 100, 8))
            pygame.draw.rect(self.screen, POWERED_COLOR, (WIDTH // 2 - 50, y + 28, max(0, bar_w), 8))
            txt = self.small.render("POWERED", True, POWERED_COLOR)
            self.screen.blit(txt, (WIDTH // 2 - 28, y + 10))
            
        TOGGLE_WIDTH, TOGGLE_HEIGHT = 160, 32
        TOGGLE_RECT = pygame.Rect(WIDTH - TOGGLE_WIDTH * 2 - 20, y + 8, TOGGLE_WIDTH, TOGGLE_HEIGHT)
        RL_TOGGLE_RECT = pygame.Rect(WIDTH - TOGGLE_WIDTH - 10, y + 8, TOGGLE_WIDTH, TOGGLE_HEIGHT)
        
        # Draw static AUTO MODE button
        bg_btn = (0, 200, 100) if self.auto_mode else GREY
        pygame.draw.rect(self.screen, bg_btn, TOGGLE_RECT, border_radius=4)
        lbl_msg = "AUTO MODE" if self.auto_mode else "MANUAL PLAY"
        text_btn = self.small.render(lbl_msg, True, WHITE)
        text_rect = text_btn.get_rect(center=TOGGLE_RECT.center)
        self.screen.blit(text_btn, text_rect)
        
        # Draw static RL MODE button
        bg_btn_rl = (200, 0, 100) if self.rl_mode else GREY
        pygame.draw.rect(self.screen, bg_btn_rl, RL_TOGGLE_RECT, border_radius=4)
        lbl_msg_rl = "RL MODE ON" if self.rl_mode else "RL MODE OFF"
        text_btn_rl = self.small.render(lbl_msg_rl, True, WHITE)
        text_rect_rl = text_btn_rl.get_rect(center=RL_TOGGLE_RECT.center)
        self.screen.blit(text_btn_rl, text_rect_rl)

    def draw_bots(self):
        for gid, pos in list(self.ghost_positions.items()):
            if pos:
                sx, sy = self._world_to_screen(pos[0], pos[1])
                x = int(sx + CELL // 2)
                y = int(sy + CELL // 2)
                r = CELL // 2 - 2
                
                color = GHOST_COLORS[gid % len(GHOST_COLORS)]
                pygame.draw.circle(self.screen, color, (x, y), r)
                pygame.draw.rect(self.screen, color, (x - r, y, r * 2, r + 2))
                
                eye_r = 3
                pygame.draw.circle(self.screen, WHITE, (x - 3, y - 2), eye_r)
                pygame.draw.circle(self.screen, WHITE, (x + 3, y - 2), eye_r)
                pygame.draw.circle(self.screen, BLUE, (x - 3, y - 2), 1)
                pygame.draw.circle(self.screen, BLUE, (x + 3, y - 2), 1)

        if self.pacman_pos:
            sx, sy = self._world_to_screen(self.pacman_pos[0], self.pacman_pos[1])
            x = int(sx + CELL // 2)
            y = int(sy + CELL // 2)
            r = CELL // 2 - 2
            angle_deg = math.degrees(self.pacman_yaw)
            screen_angle_deg = -angle_deg
            
            pac_color = POWERED_COLOR if self.powered else YELLOW
            if self.mouth_open:
                gap = 35
                start_a = math.radians(screen_angle_deg + gap)
                points = [(x, y)]
                steps = 20
                full = math.radians(360 - gap * 2)
                for i in range(steps + 1):
                    a = start_a + full * i / steps
                    points.append((x + r * math.cos(a), y - r * math.sin(a)))
                pygame.draw.polygon(self.screen, pac_color, points)
            else:
                pygame.draw.circle(self.screen, pac_color, (x, y), r)

    def draw_personal_map(self):
        ghost = self.ghost_states.get(self.debug_ghost_id)
        if not ghost:
            return
            
        for r in range(ROWS):
            for c in range(COLS):
                x = WIDTH + c * CELL
                y = r * CELL
                val = ghost.personal_map[r][c]
                if val == UNKNOWN:
                    color = (30, 30, 30)
                elif val == WALL:
                    color = BLUE
                elif val == PELLET:
                    color = (180, 180, 180)
                elif val == POWER:
                    color = (255, 200, 0)
                elif val == EMPTY:
                    color = BLACK
                else:
                    color = (30, 30, 30)
                pygame.draw.rect(self.screen, color, (x, y, CELL, CELL))
                
        if ghost.belief_cells:
            # We copy belief_cells logic to avoid dictionary changed size during iteration
            belief_copy = dict(ghost.belief_cells)
            max_p = max(belief_copy.values()) if belief_copy else 0.0
            if max_p > 1e-9:
                cell_surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                for (r, c), p in belief_copy.items():
                    if p < 0.001:
                        continue
                    t = min(1.0, p / max_p)
                    red = int(t * 255)
                    green = int((1.0 - t) * 40)
                    blue = int((1.0 - t) * 210)
                    alpha = int(60 + t * 180)
                    cell_surf.fill((red, green, blue, alpha))
                    self.screen.blit(cell_surf, (WIDTH + c * CELL, r * CELL))
                    
        for gid, pos in list(ghost.known_agents.items()):
            if pos == "UNKNOWN":
                continue
            gr, gc = pos
            x = WIDTH + gc * CELL + CELL // 2
            y = gr * CELL + CELL // 2
            pygame.draw.circle(self.screen, GHOST_COLORS[gid % len(GHOST_COLORS)], (x, y), CELL // 2 - 2)
            label = self.small.render(str(gid), True, WHITE)
            self.screen.blit(label, (WIDTH + gc * CELL + 2, gr * CELL + 2))
            
        x = WIDTH + ghost.col * CELL + CELL // 2
        y = ghost.row * CELL + CELL // 2
        pygame.draw.circle(self.screen, GHOST_COLORS[self.debug_ghost_id % len(GHOST_COLORS)], (x, y), CELL // 2 - 2)
        label = self.small.render(str(self.debug_ghost_id), True, WHITE)
        self.screen.blit(label, (WIDTH + ghost.col * CELL + 2, ghost.row * CELL + 2))
        
        if ghost.known_pacman:
            pr, pc = ghost.known_pacman
            x = WIDTH + pc * CELL + CELL // 2
            y = pr * CELL + CELL // 2
            pygame.draw.circle(self.screen, POWERED_COLOR if ghost.pacman_powered else YELLOW, (x, y), CELL // 2 - 2)
            label = self.small.render("P", True, BLACK)
            self.screen.blit(label, (WIDTH + pc * CELL + 2, pr * CELL + 2))
            
        txt = self.small.render(f"Ghost {self.debug_ghost_id} local map + belief heatmap  [0-6 to switch]", True, WHITE)
        self.screen.blit(txt, (WIDTH + 4, ROWS * CELL + 6))

    def run(self):
        while True:
            self.handle_events()
            self.update()
            
            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_bots()
            self.draw_hud()
            self.draw_personal_map()
            
            pygame.display.flip()
            self.clock.tick(FPS)

def spin_node(node):
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass

def main():
    rclpy.init()
    
    game = Game()
    ros_node = VisualizerNode(game)
    
    t = threading.Thread(target=spin_node, args=(ros_node,), daemon=True)
    t.start()
    
    try:
        game.run()
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        ros_node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
