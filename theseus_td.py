"""
Theseus & the Maze — DP/MDP + TD Learning Workshop Visualizer
==============================================================
Part 1: Value/Policy Iteration (global sweep, omniscient)
Part 2: TD Learning (local updates, fog of war, experiential)

Controls:
  ═══════════════════════════════════════════════════════════════
  MODE SWITCHING
  ═══════════════════════════════════════════════════════════════
  T           — Toggle TD Learning mode (Blind Navigator)
  TAB         — Switch between DP and TD display panels
  
  ═══════════════════════════════════════════════════════════════
  DP MODE (Value/Policy Iteration)
  ═══════════════════════════════════════════════════════════════
  SPACE       — Start / pause value iteration scan
  S           — Step the agent along learned policy
  M           — Toggle MDP (stochastic) / deterministic
  P           — Toggle Policy Iteration / Value Iteration
  G           — Cycle gamma (0.7 → 0.9 → 0.99)
  +/-         — Speed up / slow down scan
  
  ═══════════════════════════════════════════════════════════════
  TD MODE (Temporal Difference Learning)
  ═══════════════════════════════════════════════════════════════
  W/A/S/D     — Move agent (trigger TD update)
  E           — Auto-explore (ε-greedy policy)
  A           — Cycle alpha (0.05 → 0.1 → 0.3 → 0.5)
  M           — Toggle MDP mode (affects transitions)
  G           — Cycle gamma
  C           — Clear fog / reveal all cells
  
  ═══════════════════════════════════════════════════════════════
  COMMON
  ═══════════════════════════════════════════════════════════════
  R           — Reset (values, policy, fog)
  N           — New random maze
  Left-click  — Toggle wall on any cell
  Right-click — Move goal to that cell
  Middle-click— Move start to that cell
"""

import pygame
import random
import sys
import math
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from collections import deque

pygame.init()


# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
class Config:
    GRID_SIZE        = 7
    CELL_SIZE        = 90
    GRID_OFFSET_X    = 14
    GRID_OFFSET_Y    = 70
    GRID_PX          = GRID_SIZE * CELL_SIZE
    INFO_W           = 330
    WIN_W            = GRID_PX + GRID_OFFSET_X * 2 + INFO_W
    WIN_H            = GRID_PX + GRID_OFFSET_Y + 20
    
    # DP parameters
    SCAN_SPEED_MS    = 60
    SPEED_STEP       = 15
    SPEED_MIN        = 5
    SPEED_MAX        = 400
    CONVERGENCE_EPS  = 0.001
    
    # TD parameters
    ALPHAS           = [0.05, 0.1, 0.3, 0.5]
    GAMMAS           = [0.70, 0.90, 0.99]
    EPSILON          = 0.2     # for ε-greedy exploration
    TD_FLASH_FRAMES  = 20      # visual feedback duration
    
    # Rewards
    REWARD_GOAL      = 100
    REWARD_STEP      = -1
    
    # MDP stochastic
    MDP_PROB_FORWARD = 0.80
    MDP_PROB_SIDEWAYS = 0.10
    
    # ── Palette ───────────────────────────
    BG           = (18, 18, 28)
    GRID_BG      = (28, 28, 44)
    PANEL_BG     = (22, 22, 36)
    WALL         = (40, 42, 60)
    WALL_BORDER  = (60, 62, 85)
    GOAL_A       = (80, 220, 140)
    GOAL_B       = (40, 180, 100)
    START_C      = (100, 160, 255)
    SCAN_C       = (255, 210, 60)
    AGENT_C      = (255, 80, 80)
    ARROW_C      = (255, 255, 255)
    PATH_C       = (255, 100, 60)
    LINE_C       = (80, 140, 255)
    TEXT_DIM     = (120, 120, 160)
    TEXT_MID     = (180, 180, 210)
    TEXT_BRIGHT  = (240, 240, 255)
    ACCENT_VI    = (80, 140, 255)
    ACCENT_PI    = (255, 140, 80)
    ACCENT_TD    = (140, 80, 255)
    ACCENT_MDP   = (80, 220, 180)
    CONVERGED_C  = (80, 220, 140)
    DELTA_BAR_C  = (255, 80, 80)
    
    # TD specific colors
    FOG_C        = (35, 35, 50)         # unvisited cells
    VISITED_C    = (45, 48, 70)         # visited but no value yet
    TD_FLASH_C   = (255, 220, 100)      # TD update flash
    TRAJECTORY_C = (255, 150, 255)      # agent path
    TD_POS_C     = (100, 255, 140)      # positive TD error
    TD_NEG_C     = (255, 100, 140)      # negative TD error
    
    # Value heatmap
    HEAT_LOW     = (30, 20, 80)
    HEAT_HIGH    = (255, 220, 60)


# ─────────────────────────────────────────
#  Direction
# ─────────────────────────────────────────
class Direction(Enum):
    UP    = (-1, 0)
    DOWN  = ( 1, 0)
    LEFT  = ( 0, -1)
    RIGHT = ( 0, 1)

PERP = {
    Direction.UP:    (Direction.LEFT, Direction.RIGHT),
    Direction.DOWN:  (Direction.LEFT, Direction.RIGHT),
    Direction.LEFT:  (Direction.UP, Direction.DOWN),
    Direction.RIGHT: (Direction.UP, Direction.DOWN),
}

DIR_LABELS = {
    Direction.UP:    "↑", Direction.DOWN:  "↓",
    Direction.LEFT:  "←", Direction.RIGHT: "→",
}


# ─────────────────────────────────────────
#  Position
# ─────────────────────────────────────────
@dataclass
class Position:
    row: int
    col: int
    
    def __hash__(self):  return hash((self.row, self.col))
    def __eq__(self, o): return self.row == o.row and self.col == o.col
    def move(self, d: Direction): return Position(self.row + d.value[0], self.col + d.value[1])
    def valid(self): return 0 <= self.row < Config.GRID_SIZE and 0 <= self.col < Config.GRID_SIZE


# ─────────────────────────────────────────
#  MazeState
# ─────────────────────────────────────────
class MazeState:
    def __init__(self):
        self.grid_size   = Config.GRID_SIZE
        self.start_pos   = Position(0, 0)
        self.goal_pos    = Position(6, 6)
        self.agent_pos   = Position(0, 0)
        self.grid: List[List[str]] = [['.' for _ in range(Config.GRID_SIZE)]
                                       for _ in range(Config.GRID_SIZE)]
        
        # Learning parameters
        self.gamma       = Config.GAMMAS[1]      # 0.90
        self._gamma_idx  = 1
        self.alpha       = Config.ALPHAS[1]      # 0.1
        self._alpha_idx  = 1
        self.is_mdp      = False
        
        # DP state
        self.use_pi      = False                 # Policy Iteration?
        self.values: Dict[Position, float] = {}
        self.policy: Dict[Position, Direction] = {}
        self.pi_policy: Dict[Position, Direction] = {}
        self.current_scan = Position(0, 0)
        self.is_running   = False
        self.last_update_ms = 0
        self.scan_speed   = Config.SCAN_SPEED_MS
        self.iteration    = 0
        self.sweep_count  = 0
        self.max_delta    = float('inf')
        self.converged    = False
        self.delta_history: List[float] = []
        self.show_path    = False
        self.optimal_path: List[Position] = []
        
        # TD state
        self.td_mode      = False                # TD learning active?
        self.td_visited   = set()                # cells the agent has been to
        self.td_trajectory = deque(maxlen=50)    # recent agent path
        self.td_auto_explore = False             # ε-greedy auto-exploration
        self.td_episode   = 0                    # episode counter
        self.td_steps     = 0                    # steps in current episode
        self.td_last_error = 0.0                 # most recent TD error
        self.td_error_history: List[float] = []  # TD errors over time
        self.td_flash_cells: Dict[Position, int] = {}  # cell → frames_left
        self.td_prev_pos  = None                 # for TD update
        
        self._generate_default_maze()
        self.reset_values()
    
    # ── Maze setup ────────────────────────
    def _generate_default_maze(self):
        walls = [(1,1),(1,2),(1,4),(1,5),
                 (3,0),(3,1),(3,3),(3,4),
                 (5,2),(5,3),(5,4),(5,6)]
        for r, c in walls:
            self.grid[r][c] = '#'
    
    def generate_random_maze(self):
        """DFS-based random maze."""
        g = Config.GRID_SIZE
        self.grid = [['#' for _ in range(g)] for _ in range(g)]
        
        def carve(r, c):
            self.grid[r][c] = '.'
            dirs = [(0,2),(0,-2),(2,0),(-2,0)]
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if 0 <= nr < g and 0 <= nc < g and self.grid[nr][nc] == '#':
                    self.grid[r+dr//2][c+dc//2] = '.'
                    carve(nr, nc)
        
        carve(0, 0)
        self.grid[self.start_pos.row][self.start_pos.col] = '.'
        self.grid[self.goal_pos.row][self.goal_pos.col]   = '.'
        for _ in range(6):
            r, c = random.randrange(g), random.randrange(g)
            if (r,c) not in [(self.start_pos.row,self.start_pos.col),
                             (self.goal_pos.row,self.goal_pos.col)]:
                self.grid[r][c] = '.'
        self.reset_values()
    
    # ── Value / policy init ───────────────
    def reset_values(self):
        self.values   = {Position(r, c): 0.0
                         for r in range(Config.GRID_SIZE)
                         for c in range(Config.GRID_SIZE)}
        self.values[self.goal_pos] = Config.REWARD_GOAL
        self.policy   = {}
        self.pi_policy = {}
        
        # DP reset
        self.current_scan = Position(0, 0)
        self.is_running  = False
        self.iteration   = 0
        self.sweep_count = 0
        self.max_delta   = float('inf')
        self.converged   = False
        self.delta_history = []
        self.show_path   = False
        self.optimal_path = []
        
        # TD reset
        self.td_visited   = set()
        self.td_trajectory = deque(maxlen=50)
        self.td_auto_explore = False
        self.td_episode   = 0
        self.td_steps     = 0
        self.td_last_error = 0.0
        self.td_error_history = []
        self.td_flash_cells = {}
        self.td_prev_pos  = None
        
        self.agent_pos   = Position(self.start_pos.row, self.start_pos.col)
        
        # Random initial policy for PI
        for r in range(Config.GRID_SIZE):
            for c in range(Config.GRID_SIZE):
                pos = Position(r, c)
                if self.grid[r][c] != '#' and pos != self.goal_pos:
                    neighbors = self.get_valid_neighbors(pos)
                    if neighbors:
                        self.pi_policy[pos] = random.choice(neighbors)[1]
    
    def cycle_gamma(self):
        self._gamma_idx = (self._gamma_idx + 1) % len(Config.GAMMAS)
        self.gamma = Config.GAMMAS[self._gamma_idx]
        if not self.td_mode:
            self.reset_values()
    
    def cycle_alpha(self):
        self._alpha_idx = (self._alpha_idx + 1) % len(Config.ALPHAS)
        self.alpha = Config.ALPHAS[self._alpha_idx]
    
    # ── Transition model ──────────────────
    def get_valid_neighbors(self, pos: Position):
        result = []
        for d in Direction:
            nxt = pos.move(d)
            if nxt.valid() and self.grid[nxt.row][nxt.col] != '#':
                result.append((nxt, d))
        return result
    
    def transitions(self, pos: Position, action: Direction):
        """Returns list of (probability, next_position)."""
        if not self.is_mdp:
            nxt = pos.move(action)
            if not nxt.valid() or self.grid[nxt.row][nxt.col] == '#':
                nxt = pos
            return [(1.0, nxt)]
        
        outcomes: Dict[Position, float] = {}
        
        def add(prob, direction):
            nxt = pos.move(direction)
            if not nxt.valid() or self.grid[nxt.row][nxt.col] == '#':
                nxt = pos
            outcomes[nxt] = outcomes.get(nxt, 0) + prob
        
        add(Config.MDP_PROB_FORWARD, action)
        for perp_dir in PERP[action]:
            add(Config.MDP_PROB_SIDEWAYS, perp_dir)

        # outcomes maps next_position -> probability; callers expect (probability, next_position)
        return [(prob, nxt) for nxt, prob in outcomes.items()]
    
    def bellman_value(self, pos: Position, action: Direction) -> float:
        """Q(s, a) = Σ p(s'|s,a) [r + γ V(s')]"""
        total = 0.0
        for prob, nxt in self.transitions(pos, action):
            r = Config.REWARD_GOAL if nxt == self.goal_pos else Config.REWARD_STEP
            total += prob * (r + self.gamma * self.values[nxt])
        return total
    
    # ═══════════════════════════════════════
    #  DP Methods
    # ═══════════════════════════════════════
    def update_single_cell(self, pos: Position) -> float:
        """DP update for one cell. Returns |delta|."""
        if self.grid[pos.row][pos.col] == '#' or pos == self.goal_pos:
            return 0.0
        
        neighbors = self.get_valid_neighbors(pos)
        if not neighbors:
            return 0.0
        
        old_v = self.values[pos]
        
        if self.use_pi:
            action = self.pi_policy.get(pos, neighbors[0][1])
            new_v  = self.bellman_value(pos, action)
            self.values[pos] = new_v
            best_v, best_d = max(
                ((self.bellman_value(pos, d), d) for _, d in neighbors),
                key=lambda x: x[0]
            )
            self.pi_policy[pos] = best_d
            self.policy[pos]    = best_d
        else:
            best_v, best_d = max(
                ((self.bellman_value(pos, d), d) for _, d in neighbors),
                key=lambda x: x[0]
            )
            self.values[pos] = best_v
            self.policy[pos] = best_d
        
        return abs(self.values[pos] - old_v)
    
    def advance_scan(self):
        """Advance DP scan one cell."""
        delta = self.update_single_cell(self.current_scan)
        self.iteration += 1
        
        if not hasattr(self, '_sweep_delta'):
            self._sweep_delta = 0.0
        self._sweep_delta = max(self._sweep_delta, delta)
        
        c = self.current_scan.col + 1
        r = self.current_scan.row
        if c >= Config.GRID_SIZE:
            c = 0; r += 1
        sweep_done = False
        if r >= Config.GRID_SIZE:
            r = 0; c = 0
            self.sweep_count  += 1
            self.max_delta     = self._sweep_delta
            self._sweep_delta  = 0.0
            sweep_done         = True
            self.delta_history.append(min(self.max_delta, Config.REWARD_GOAL))
            if len(self.delta_history) > 30:
                self.delta_history.pop(0)
            if self.max_delta < Config.CONVERGENCE_EPS:
                self.converged  = True
                self.is_running = False
                self.compute_optimal_path()
        
        self.current_scan = Position(r, c)
        return sweep_done
    
    def compute_optimal_path(self):
        path = [Position(self.start_pos.row, self.start_pos.col)]
        visited = {path[0]}
        for _ in range(Config.GRID_SIZE ** 2):
            cur = path[-1]
            if cur == self.goal_pos:
                break
            if cur not in self.policy:
                break
            nxt = cur.move(self.policy[cur])
            if nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
        self.optimal_path = path
        self.show_path = True
    
    def step_agent(self):
        """Step agent along policy (for DP mode)."""
        if self.agent_pos == self.goal_pos:
            self.agent_pos = Position(self.start_pos.row, self.start_pos.col)
            return
        if self.agent_pos in self.policy:
            action = self.policy[self.agent_pos]
            outcomes = self.transitions(self.agent_pos, action)
            roll = random.random()
            cumul = 0.0
            chosen = self.agent_pos
            for prob, nxt in outcomes:
                cumul += prob
                if roll <= cumul:
                    chosen = nxt
                    break
            self.agent_pos = chosen
    
    # ═══════════════════════════════════════
    #  TD Methods (Human Mode)
    # ═══════════════════════════════════════
    def td_move(self, direction: Direction) -> bool:
        """
        Move agent in TD mode, perform TD update.
        Returns True if goal reached.
        """
        if self.agent_pos == self.goal_pos:
            return True
        
        # Store previous position for TD update
        prev_pos = Position(self.agent_pos.row, self.agent_pos.col)
        
        # Sample transition (stochastic if MDP)
        outcomes = self.transitions(self.agent_pos, direction)
        roll = random.random()
        cumul = 0.0
        next_pos = self.agent_pos
        for prob, nxt in outcomes:
            cumul += prob
            if roll <= cumul:
                next_pos = nxt
                break
        
        # Get reward
        reward = Config.REWARD_GOAL if next_pos == self.goal_pos else Config.REWARD_STEP
        
        # TD Update: V(s) ← V(s) + α[r + γV(s') - V(s)]
        td_target = reward + self.gamma * self.values[next_pos]
        td_error  = td_target - self.values[prev_pos]
        self.values[prev_pos] += self.alpha * td_error
        
        # Track
        self.td_last_error = td_error
        self.td_error_history.append(td_error)
        if len(self.td_error_history) > 100:
            self.td_error_history.pop(0)
        
        # Visual flash
        self.td_flash_cells[prev_pos] = Config.TD_FLASH_FRAMES
        
        # Update policy greedily at this cell
        self._update_policy_at_cell(prev_pos)
        
        # Move agent
        self.agent_pos = next_pos
        self.td_visited.add(next_pos)
        self.td_trajectory.append(next_pos)
        self.td_steps += 1
        
        # Check goal
        if next_pos == self.goal_pos:
            self.td_episode += 1
            return True
        
        return False
    
    def _update_policy_at_cell(self, pos: Position):
        """Greedy policy update for one cell."""
        if pos == self.goal_pos or self.grid[pos.row][pos.col] == '#':
            return
        neighbors = self.get_valid_neighbors(pos)
        if not neighbors:
            return
        best_v, best_d = max(
            ((self.values[nxt], d) for nxt, d in neighbors),
            key=lambda x: x[0]
        )
        self.policy[pos] = best_d
    
    def td_reset_episode(self):
        """Reset agent to start, keep learned values."""
        self.agent_pos = Position(self.start_pos.row, self.start_pos.col)
        self.td_trajectory = deque(maxlen=50)
        self.td_steps = 0
    
    def td_auto_step(self):
        """
        Epsilon-greedy exploration step.
        """
        if self.agent_pos == self.goal_pos:
            self.td_reset_episode()
            return
        
        # Epsilon-greedy action selection
        neighbors = self.get_valid_neighbors(self.agent_pos)
        if not neighbors:
            return
        
        if random.random() < Config.EPSILON or self.agent_pos not in self.policy:
            # Explore: random action
            _, direction = random.choice(neighbors)
        else:
            # Exploit: follow policy
            direction = self.policy[self.agent_pos]
        
        reached_goal = self.td_move(direction)
        if reached_goal:
            self.td_reset_episode()
    
    def td_reveal_all(self):
        """Clear fog of war (cheat)."""
        for r in range(Config.GRID_SIZE):
            for c in range(Config.GRID_SIZE):
                pos = Position(r, c)
                if self.grid[r][c] != '#':
                    self.td_visited.add(pos)
    
    # ── Interaction ───────────────────────
    def toggle_wall(self, pos: Position):
        if pos == self.start_pos or pos == self.goal_pos:
            return
        self.grid[pos.row][pos.col] = '.' if self.grid[pos.row][pos.col] == '#' else '#'
        self.reset_values()
    
    def move_goal(self, pos: Position):
        if self.grid[pos.row][pos.col] != '#' and pos != self.start_pos:
            self.goal_pos = pos
            self.reset_values()
    
    def move_start(self, pos: Position):
        if self.grid[pos.row][pos.col] != '#' and pos != self.goal_pos:
            self.start_pos = pos
            self.agent_pos = Position(pos.row, pos.col)
            self.reset_values()


# ─────────────────────────────────────────
#  Renderer
# ─────────────────────────────────────────
class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font_tiny   = pygame.font.Font(None, 16)
        self.font_sm     = pygame.font.Font(None, 19)
        self.font_md     = pygame.font.Font(None, 24)
        self.font_lg     = pygame.font.Font(None, 32)
        self.font_xl     = pygame.font.Font(None, 42)
    
    def lerp_color(self, a, b, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
    
    def text(self, txt, font, color, x, y, anchor="topleft"):
        surf = font.render(str(txt), True, color)
        r    = surf.get_rect(**{anchor: (x, y)})
        self.screen.blit(surf, r)
        return r
    
    def pill(self, label, x, y, color, font=None):
        font  = font or self.font_sm
        surf  = font.render(label, True, (240, 240, 255))
        pad   = 6
        rect  = pygame.Rect(x - pad, y - pad//2, surf.get_width() + pad*2, surf.get_height() + pad)
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        self.screen.blit(surf, (x, y))
        return rect
    
    # ── Main draw ─────────────────────────
    def draw(self, state: MazeState, tick: int):
        W, H = self.screen.get_size()
        self.screen.fill(Config.BG)
        
        # ── Title bar ──────────────────────
        pygame.draw.rect(self.screen, Config.PANEL_BG, (0, 0, W, 55))
        
        if state.td_mode:
            title = "THESEUS:  THE  BLIND  NAVIGATOR"
            mode_label = "TD Learning · Experiential Updates"
            color = Config.ACCENT_TD
        else:
            title = "THESEUS:  THE  OMNISCIENT"
            mode_label = ("Policy Iteration" if state.use_pi else "Value Iteration") + \
                         (" · MDP 80/10/10" if state.is_mdp else " · Deterministic")
            color = Config.ACCENT_PI if state.use_pi else Config.ACCENT_VI
        
        self.text(title, self.font_xl, Config.TEXT_BRIGHT, 16, 10)
        self.text(mode_label, self.font_sm, Config.TEXT_DIM, 16, 40)
        
        # Mode indicator
        mode_txt = "TD MODE" if state.td_mode else "DP MODE"
        self.pill(mode_txt, W - Config.INFO_W - 100, 16, color)
        
        # ── Grid ───────────────────────────
        self.draw_grid(state, tick)
        
        # ── Info panel ─────────────────────
        if state.td_mode:
            self.draw_td_panel(state, tick)
        else:
            self.draw_dp_panel(state, tick)
    
    # ── Grid rendering ────────────────────
    def draw_grid(self, state: MazeState, tick: int):
        gx, gy = Config.GRID_OFFSET_X, Config.GRID_OFFSET_Y
        gw = gh = Config.GRID_PX
        pygame.draw.rect(self.screen, Config.GRID_BG, (gx - 4, gy - 4, gw + 8, gh + 8), border_radius=6)
        
        # Compute value range
        if state.td_mode:
            # Only consider visited cells
            vals = [state.values[p] for p in state.td_visited
                    if state.grid[p.row][p.col] != '#' and p != state.goal_pos]
        else:
            vals = [v for p, v in state.values.items()
                    if state.grid[p.row][p.col] != '#' and p != state.goal_pos]
        
        v_min = min(vals) if vals else 0
        v_max = max(vals) if vals else 1
        v_range = max(v_max - v_min, 1e-6)
        
        cs = Config.CELL_SIZE
        
        # ── Draw cells ─────────────────────
        for r in range(Config.GRID_SIZE):
            for c in range(Config.GRID_SIZE):
                pos  = Position(r, c)
                cx   = gx + c * cs
                cy   = gy + r * cs
                rect = pygame.Rect(cx, cy, cs, cs)
                
                # Cell color
                if state.grid[r][c] == '#':
                    color = Config.WALL
                    pygame.draw.rect(self.screen, color, rect)
                    # Hatch
                    for i in range(0, cs + cs, 16):
                        pygame.draw.line(self.screen, Config.WALL_BORDER,
                                         (cx + i, cy), (cx, cy + i), 1)
                
                elif pos == state.goal_pos:
                    pulse = 0.5 + 0.5 * math.sin(tick * 0.006)
                    color = self.lerp_color(Config.GOAL_B, Config.GOAL_A, pulse)
                    pygame.draw.rect(self.screen, color, rect)
                
                elif state.td_mode and pos not in state.td_visited:
                    # Fog of war
                    pygame.draw.rect(self.screen, Config.FOG_C, rect)
                
                elif pos in state.td_flash_cells:
                    # TD update flash
                    flash_t = state.td_flash_cells[pos] / Config.TD_FLASH_FRAMES
                    color = self.lerp_color(Config.HEAT_HIGH, Config.TD_FLASH_C, flash_t)
                    pygame.draw.rect(self.screen, color, rect)
                
                elif pos == state.current_scan and state.is_running and not state.td_mode:
                    pygame.draw.rect(self.screen, Config.SCAN_C, rect)
                
                else:
                    # Heatmap
                    v = state.values[pos]
                    t = (v - v_min) / v_range
                    color = self.lerp_color(Config.HEAT_LOW, Config.HEAT_HIGH, t)
                    pygame.draw.rect(self.screen, color, rect)
                
                # Border
                border_c = (60, 62, 85) if state.grid[r][c] == '#' else (50, 52, 75)
                pygame.draw.rect(self.screen, border_c, rect, 1)
                
                # Value text
                if state.grid[r][c] != '#':
                    if state.td_mode and pos not in state.td_visited:
                        # Hidden
                        self.text("?", self.font_md, (100, 100, 140), cx + cs//2, cy + cs//2, anchor="center")
                    else:
                        v = state.values[pos]
                        v_color = (255, 255, 255) if pos == state.goal_pos else \
                                  (200, 200, 230) if v > (v_min + v_range * 0.5) else (140, 140, 180)
                        if pos == state.goal_pos:
                            self.text(f"+{Config.REWARD_GOAL}", self.font_md, v_color, cx + 5, cy + 5)
                        else:
                            self.text(f"{v:.1f}", self.font_sm, v_color, cx + 5, cy + 5)
                        
                        # Arrow
                        if pos in state.policy and pos != state.goal_pos:
                            if not state.td_mode or pos in state.td_visited:
                                self.draw_arrow(cx + cs//2, cy + cs//2,
                                                state.policy[pos], cs // 2 - 8)
        
        # ── TD trajectory ──────────────────
        if state.td_mode and len(state.td_trajectory) > 1:
            pts = [(gx + p.col * cs + cs // 2, gy + p.row * cs + cs // 2)
                   for p in state.td_trajectory]
            pygame.draw.lines(self.screen, Config.TRAJECTORY_C, False, pts, 2)
        
        # ── DP lookahead lines ─────────────
        if state.is_running and not state.td_mode:
            ox = gx + state.current_scan.col * cs + cs // 2
            oy = gy + state.current_scan.row * cs + cs // 2
            for nxt, _ in state.get_valid_neighbors(state.current_scan):
                nx = gx + nxt.col * cs + cs // 2
                ny = gy + nxt.row * cs + cs // 2
                pygame.draw.line(self.screen, Config.LINE_C, (ox, oy), (nx, ny), 2)
        
        # ── Optimal path ───────────────────
        if state.show_path and len(state.optimal_path) > 1:
            pts = [(gx + p.col * cs + cs // 2, gy + p.row * cs + cs // 2)
                   for p in state.optimal_path]
            pygame.draw.lines(self.screen, Config.PATH_C, False, pts, 3)
            for pt in pts:
                pygame.draw.circle(self.screen, Config.PATH_C, pt, 5)
        
        # ── Start marker ───────────────────
        sx = gx + state.start_pos.col * cs + cs // 2
        sy = gy + state.start_pos.row * cs + cs // 2
        pygame.draw.circle(self.screen, Config.START_C, (sx, sy), 8)
        pygame.draw.circle(self.screen, Config.BG, (sx, sy), 4)
        
        # ── Agent ──────────────────────────
        ax = gx + state.agent_pos.col * cs + cs // 2
        ay = gy + state.agent_pos.row * cs + cs // 2
        pulse = 0.5 + 0.5 * math.sin(tick * 0.008)
        r_agent = int(12 + 3 * pulse)
        pygame.draw.circle(self.screen, Config.BG, (ax, ay), r_agent + 2)
        pygame.draw.circle(self.screen, Config.AGENT_C, (ax, ay), r_agent)
        pygame.draw.circle(self.screen, (255, 200, 200), (ax, ay), r_agent - 4)
    
    def draw_arrow(self, cx, cy, d: Direction, size: int):
        dr, dc = d.value
        tip   = (cx + dc * size, cy + dr * size)
        left  = (cx - dr * size // 3, cy + dc * size // 3)
        right = (cx + dr * size // 3, cy - dc * size // 3)
        pygame.draw.polygon(self.screen, Config.ARROW_C, [tip, left, (cx, cy), right], 0)
        pygame.draw.polygon(self.screen, (*Config.ARROW_C[:3],), [tip, left, right], 1)
    
    # ── DP Info Panel ─────────────────────
    def draw_dp_panel(self, state: MazeState, tick: int):
        W, H = self.screen.get_size()
        px   = W - Config.INFO_W + 8
        py   = Config.GRID_OFFSET_Y
        iw   = Config.INFO_W - 16
        line = 0
        
        def nl(n=1): nonlocal line; line += n
        def heading(txt, color=Config.TEXT_BRIGHT):
            self.text(txt, self.font_md, color, px, py + line)
            nl(22)
        def row(label, val, val_color=Config.TEXT_MID):
            self.text(label, self.font_sm, Config.TEXT_DIM, px, py + line)
            self.text(str(val), self.font_sm, val_color, px + iw, py + line, anchor="topright")
            nl(20)
        def divider():
            pygame.draw.line(self.screen, Config.WALL_BORDER, (px, py + line), (px + iw, py + line))
            nl(10)
        
        # Status
        if state.converged:
            self.pill(" ✓ CONVERGED ", px, py + line, Config.CONVERGED_C, self.font_md)
            nl(34)
        elif state.is_running:
            pulse = 0.5 + 0.5 * math.sin(tick * 0.012)
            c = self.lerp_color(Config.ACCENT_VI, (180, 220, 255), pulse)
            self.pill(" ▶ RUNNING ", px, py + line, c, self.font_md)
            nl(34)
        else:
            self.pill(" ■ PAUSED ", px, py + line, Config.TEXT_DIM, self.font_md)
            nl(34)
        divider()
        
        # Algorithm params
        heading("Algorithm")
        alg = "Policy Iteration" if state.use_pi else "Value Iteration"
        algo_c = Config.ACCENT_PI if state.use_pi else Config.ACCENT_VI
        row("Method", alg, algo_c)
        row("Environment", "MDP (stoch.)" if state.is_mdp else "Deterministic",
            Config.ACCENT_MDP if state.is_mdp else Config.TEXT_MID)
        row("γ (discount)", f"{state.gamma:.2f}")
        row("Iterations", state.iteration)
        row("Sweeps", state.sweep_count)
        delta_str = f"{state.max_delta:.4f}" if state.max_delta < 1e9 else "—"
        delta_c = Config.CONVERGED_C if state.max_delta < Config.CONVERGENCE_EPS else Config.DELTA_BAR_C
        row("Max Δ (sweep)", delta_str, delta_c)
        row("Speed", f"{state.scan_speed} ms/cell")
        nl(4)
        divider()
        
        # Bellman
        heading("Bellman Equation")
        eq_lines = [
            "V(s) = max_a Σ p(s'|s,a)",
            "          · [r + γ V(s')]",
        ]
        if state.is_mdp:
            eq_lines.append("")
            eq_lines.append(f"p(fwd)={Config.MDP_PROB_FORWARD:.0%}")
            eq_lines.append(f"p(side)={Config.MDP_PROB_SIDEWAYS:.0%}")
        for el in eq_lines:
            self.text(el, self.font_tiny, (160, 200, 255), px, py + line)
            nl(16)
        nl(4)
        divider()
        
        # Convergence sparkline
        heading("Convergence (max Δ)")
        hist = state.delta_history
        if hist:
            bw = iw // max(len(hist), 1)
            bw = max(bw, 2)
            bh = 40
            bx0 = px
            by0 = py + line
            h_max = max(hist) if max(hist) > 0 else 1
            for i, dv in enumerate(hist):
                hh = int((dv / h_max) * bh)
                bx = bx0 + i * bw
                by = by0 + bh - hh
                t = dv / (h_max + 1e-6)
                bc = self.lerp_color(Config.CONVERGED_C, Config.DELTA_BAR_C, t)
                pygame.draw.rect(self.screen, bc, (bx, by, bw - 1, hh))
            pygame.draw.rect(self.screen, Config.WALL_BORDER, (bx0, by0, iw, bh), 1)
            nl(bh + 8)
        else:
            self.text("(no data yet)", self.font_sm, Config.TEXT_DIM, px, py + line)
            nl(24)
        divider()
        
        # Controls
        heading("DP Controls")
        controls = [
            ("T",          "→ TD mode"),
            ("SPACE",      "Start/pause"),
            ("S",          "Step agent"),
            ("M",          "Toggle MDP"),
            ("P",          "Toggle PI/VI"),
            ("G",          "Cycle gamma"),
            ("+/-",        "Speed"),
        ]
        for key, desc in controls:
            key_surf = self.font_tiny.render(key, True, (255, 220, 100))
            self.screen.blit(key_surf, (px, py + line))
            self.text(desc, self.font_tiny, Config.TEXT_DIM, px + 50, py + line)
            nl(17)
    
    # ── TD Info Panel ─────────────────────
    def draw_td_panel(self, state: MazeState, tick: int):
        W, H = self.screen.get_size()
        px   = W - Config.INFO_W + 8
        py   = Config.GRID_OFFSET_Y
        iw   = Config.INFO_W - 16
        line = 0
        
        def nl(n=1): nonlocal line; line += n
        def heading(txt, color=Config.TEXT_BRIGHT):
            self.text(txt, self.font_md, color, px, py + line)
            nl(22)
        def row(label, val, val_color=Config.TEXT_MID):
            self.text(label, self.font_sm, Config.TEXT_DIM, px, py + line)
            self.text(str(val), self.font_sm, val_color, px + iw, py + line, anchor="topright")
            nl(20)
        def divider():
            pygame.draw.line(self.screen, Config.WALL_BORDER, (px, py + line), (px + iw, py + line))
            nl(10)
        
        # Status
        if state.td_auto_explore:
            pulse = 0.5 + 0.5 * math.sin(tick * 0.012)
            c = self.lerp_color(Config.ACCENT_TD, (200, 140, 255), pulse)
            self.pill(" ▶ EXPLORING ", px, py + line, c, self.font_md)
            nl(34)
        else:
            self.pill(" ■ MANUAL ", px, py + line, Config.TEXT_DIM, self.font_md)
            nl(34)
        divider()
        
        # TD params
        heading("TD Learning")
        row("α (learning rate)", f"{state.alpha:.2f}", Config.ACCENT_TD)
        row("γ (discount)", f"{state.gamma:.2f}")
        row("Environment", "MDP (stoch.)" if state.is_mdp else "Deterministic",
            Config.ACCENT_MDP if state.is_mdp else Config.TEXT_MID)
        row("ε (exploration)", f"{Config.EPSILON:.2f}")
        row("Episodes", state.td_episode)
        row("Steps (episode)", state.td_steps)
        row("Cells visited", len(state.td_visited))
        nl(4)
        divider()
        
        # TD Update display
        heading("Last TD Update")
        td_err = state.td_last_error
        td_str = f"{td_err:+.2f}"
        td_color = Config.TD_POS_C if td_err > 0 else Config.TD_NEG_C if td_err < 0 else Config.TEXT_MID
        row("TD Error", td_str, td_color)
        
        # TD equation
        self.text("V(s) ← V(s) + α·δ", self.font_sm, (160, 200, 255), px, py + line)
        nl(18)
        self.text("δ = r + γV(s') - V(s)", self.font_tiny, (140, 180, 235), px, py + line)
        nl(18)
        nl(4)
        divider()
        
        # TD error sparkline
        heading("TD Error History")
        hist = state.td_error_history
        if hist:
            bw = iw // max(len(hist), 1)
            bw = max(bw, 2)
            bh = 50
            bx0 = px
            by0 = py + line + bh // 2
            h_max = max(abs(e) for e in hist) if hist else 1
            h_max = max(h_max, 1)
            
            # Zero line
            pygame.draw.line(self.screen, Config.WALL_BORDER, (bx0, by0), (bx0 + iw, by0), 1)
            
            for i, err in enumerate(hist):
                hh = int((abs(err) / h_max) * (bh // 2))
                bx = bx0 + i * bw
                if err >= 0:
                    by = by0 - hh
                    bc = Config.TD_POS_C
                else:
                    by = by0
                    bc = Config.TD_NEG_C
                pygame.draw.rect(self.screen, bc, (bx, by, bw - 1, hh))
            
            pygame.draw.rect(self.screen, Config.WALL_BORDER, (bx0, py + line, iw, bh), 1)
            nl(bh + 8)
        else:
            self.text("(no data yet)", self.font_sm, Config.TEXT_DIM, px, py + line)
            nl(24)
        divider()
        
        # Workshop insight
        # heading("Workshop Question", Config.ACCENT_TD)
        # insight = [
        #     '"How does Theseus learn',
        #     'without seeing the whole',
        #     'maze?"',
        #     '',
        #     '→ TD Error propagates',
        #     '  value backwards from',
        #     '  the goal through',
        #     '  experience.',
        # ]
        # for txt in insight:
        #     self.text(txt, self.font_tiny, Config.TEXT_DIM, px, py + line)
        #     nl(15)
        # nl(4)
        # divider()
        
        # Controls
        heading("TD Controls")
        controls = [
            ("T",          "→ DP mode"),
            ("W/A/S/D",    "Move agent"),
            ("E",          "Auto-explore"),
            ("A",          "Cycle alpha"),
            ("G",          "Cycle gamma"),
            ("M",          "Toggle MDP"),
            ("C",          "Reveal all"),
            ("R",          "Reset"),
        ]
        for key, desc in controls:
            key_surf = self.font_tiny.render(key, True, (255, 220, 100))
            self.screen.blit(key_surf, (px, py + line))
            self.text(desc, self.font_tiny, Config.TEXT_DIM, px + 70, py + line)
            nl(17)


# ─────────────────────────────────────────
#  App
# ─────────────────────────────────────────
class App:
    def __init__(self):
        self.screen = pygame.display.set_mode((Config.WIN_W, Config.WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption("Theseus & the Maze — DP/TD Workshop")
        self.state = MazeState()
        self.renderer = Renderer(self.screen)
        self.clock = pygame.time.Clock()
        self.tick = 0
    
    def cell_from_mouse(self, mx, my) -> Optional[Position]:
        c = (mx - Config.GRID_OFFSET_X) // Config.CELL_SIZE
        r = (my - Config.GRID_OFFSET_Y) // Config.CELL_SIZE
        if 0 <= r < Config.GRID_SIZE and 0 <= c < Config.GRID_SIZE:
            return Position(r, c)
        return None
    
    def run(self):
        while True:
            now = pygame.time.get_ticks()
            
            # ── DP auto-scan ───────────────
            if not self.state.td_mode and self.state.is_running and \
               now - self.state.last_update_ms >= self.state.scan_speed:
                self.state.advance_scan()
                self.state.last_update_ms = now
            
            # ── TD auto-explore ────────────
            if self.state.td_mode and self.state.td_auto_explore:
                self.state.td_auto_step()
            
            # ── Decay flash effects ────────
            for pos in list(self.state.td_flash_cells.keys()):
                self.state.td_flash_cells[pos] -= 1
                if self.state.td_flash_cells[pos] <= 0:
                    del self.state.td_flash_cells[pos]
            
            # ── Draw ───────────────────────
            self.renderer.draw(self.state, self.tick)
            pygame.display.flip()
            self.tick += 1
            self.clock.tick(60)
            
            # ── Events ─────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                
                if event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    self.renderer.screen = self.screen
                
                if event.type == pygame.KEYDOWN:
                    self.on_key(event.key)
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = self.cell_from_mouse(*event.pos)
                    if pos:
                        if event.button == 1:   self.state.toggle_wall(pos)
                        elif event.button == 3: self.state.move_goal(pos)
                        elif event.button == 2: self.state.move_start(pos)
    
    def on_key(self, key):
        s = self.state
        
        # ── Mode switch ────────────────────
        if key == pygame.K_t:
            s.td_mode = not s.td_mode
            if s.td_mode:
                s.is_running = False  # stop DP scan
                s.td_visited.add(s.agent_pos)
            return
        
        # ── Common controls ────────────────
        if key == pygame.K_r:
            s.reset_values()
        
        elif key == pygame.K_n:
            s.generate_random_maze()
        
        elif key == pygame.K_m:
            s.is_mdp = not s.is_mdp
            if not s.td_mode:
                s.reset_values()
        
        elif key == pygame.K_g:
            s.cycle_gamma()
        
        # ── DP mode controls ───────────────
        elif not s.td_mode:
            if key == pygame.K_SPACE:
                if s.converged:
                    s.reset_values()
                s.is_running = not s.is_running
            
            elif key == pygame.K_s:
                s.step_agent()
            
            elif key == pygame.K_p:
                s.use_pi = not s.use_pi
                s.reset_values()
            
            elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                s.scan_speed = max(Config.SPEED_MIN, s.scan_speed - Config.SPEED_STEP)
            
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                s.scan_speed = min(Config.SPEED_MAX, s.scan_speed + Config.SPEED_STEP)
        
        # ── TD mode controls ───────────────
        else:
            if key == pygame.K_w:
                s.td_move(Direction.UP)
            elif key == pygame.K_a:
                s.td_move(Direction.LEFT)
            elif key == pygame.K_s:
                s.td_move(Direction.DOWN)
            elif key == pygame.K_d:
                s.td_move(Direction.RIGHT)
            
            elif key == pygame.K_e:
                s.td_auto_explore = not s.td_auto_explore
            
            elif key == pygame.K_a:
                s.cycle_alpha()
            
            elif key == pygame.K_c:
                s.td_reveal_all()


if __name__ == "__main__":
    App().run()