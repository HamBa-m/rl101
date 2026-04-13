"""
Theseus & the Maze — DP/MDP Workshop Visualizer
================================================
A teaching tool for Dynamic Programming and Markov Decision Processes.

Controls:
  SPACE       — Start / pause value iteration scan
  S           — Step the agent along its learned policy
  R           — Reset values (keep maze)
  N           — New random maze
  M           — Toggle MDP (stochastic) / deterministic mode
  P           — Toggle Policy Iteration / Value Iteration
  +/-         — Speed up / slow down scan
  G           — Change gamma (0.7 → 0.9 → 0.99 → 0.7)
  Left-click  — Toggle wall on any cell
  Right-click — Move goal to that cell
  Middle-click— Move start to that cell
"""

import pygame
import random
import sys
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

# ─────────────────────────────────────────
#  Bootstrap pygame
# ─────────────────────────────────────────
pygame.init()
try:
    pygame.font.Font(None, 12)          # sanity-check default font
except:
    pass

# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
class Config:
    GRID_SIZE        = 7
    CELL_SIZE        = 90
    GRID_OFFSET_X    = 14
    GRID_OFFSET_Y    = 70
    GRID_PX          = GRID_SIZE * CELL_SIZE
    INFO_W           = 310
    WIN_W            = GRID_PX + GRID_OFFSET_X * 2 + INFO_W
    WIN_H            = GRID_PX + GRID_OFFSET_Y + 20
    SCAN_SPEED_MS    = 60      # initial ms between cell updates
    SPEED_STEP       = 15
    SPEED_MIN        = 5
    SPEED_MAX        = 400
    CONVERGENCE_EPS  = 0.001

    # Rewards
    REWARD_GOAL  = 100
    REWARD_WALL  = None        # walls are impassable
    REWARD_STEP  = -1

    # MDP stochastic parameters
    MDP_PROB_FORWARD    = 0.80
    MDP_PROB_SIDEWAYS   = 0.10   # each perpendicular direction

    # Gammas to cycle through
    GAMMAS = [0.70, 0.90, 0.99]

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
    AGENT_C      = (255, 80,  80)
    ARROW_C      = (255, 255, 255)
    PATH_C       = (255, 100, 60)
    LINE_C       = (80, 140, 255)
    TEXT_DIM     = (120, 120, 160)
    TEXT_MID     = (180, 180, 210)
    TEXT_BRIGHT  = (240, 240, 255)
    ACCENT_VI    = (80, 140, 255)
    ACCENT_PI    = (255, 140, 80)
    ACCENT_MDP   = (80, 220, 180)
    CONVERGED_C  = (80, 220, 140)
    DELTA_BAR_C  = (255, 80, 80)

    # ── Value heatmap: cold→hot ────────────
    # Low value = purple-blue, high value = warm yellow
    HEAT_LOW   = (30,  20,  80)
    HEAT_HIGH  = (255, 220, 60)


# ─────────────────────────────────────────
#  Direction
# ─────────────────────────────────────────
class Direction(Enum):
    UP    = (-1, 0)
    DOWN  = ( 1, 0)
    LEFT  = ( 0,-1)
    RIGHT = ( 0, 1)

PERP = {
    Direction.UP:    (Direction.LEFT,  Direction.RIGHT),
    Direction.DOWN:  (Direction.LEFT,  Direction.RIGHT),
    Direction.LEFT:  (Direction.UP,    Direction.DOWN),
    Direction.RIGHT: (Direction.UP,    Direction.DOWN),
}

DIR_LABELS = {
    Direction.UP:    "↑ UP",
    Direction.DOWN:  "↓ DOWN",
    Direction.LEFT:  "← LEFT",
    Direction.RIGHT: "→ RIGHT",
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

    def move(self, d: Direction) -> "Position":
        return Position(self.row + d.value[0], self.col + d.value[1])

    def valid(self) -> bool:
        return 0 <= self.row < Config.GRID_SIZE and 0 <= self.col < Config.GRID_SIZE


# ─────────────────────────────────────────
#  MazeState — all model logic here
# ─────────────────────────────────────────
class MazeState:
    def __init__(self):
        self.grid_size   = Config.GRID_SIZE
        self.start_pos   = Position(0, 0)
        self.goal_pos    = Position(6, 6)
        self.agent_pos   = Position(0, 0)
        self.grid: List[List[str]] = [['.' for _ in range(Config.GRID_SIZE)]
                                       for _ in range(Config.GRID_SIZE)]
        self.gamma       = Config.GAMMAS[1]      # 0.90 default
        self._gamma_idx  = 1
        self.is_mdp      = False                 # stochastic?
        self.use_pi      = False                 # Policy Iteration?

        # DP state
        self.values: Dict[Position, float] = {}
        self.policy: Dict[Position, Direction] = {}
        self.pi_policy: Dict[Position, Direction] = {}    # locked policy for PI

        # Scan state
        self.current_scan = Position(0, 0)
        self.is_running   = False
        self.last_update_ms = 0
        self.scan_speed   = Config.SCAN_SPEED_MS
        self.iteration    = 0
        self.sweep_count  = 0         # full grid passes
        self.max_delta    = float('inf')
        self.converged    = False
        self.delta_history: List[float] = []

        # Path display
        self.show_path    = False
        self.optimal_path: List[Position] = []

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
        """DFS-based random maze with guaranteed path."""
        g = Config.GRID_SIZE
        self.grid = [['#' for _ in range(g)] for _ in range(g)]

        # Carve passages via DFS on odd cells
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
        # Ensure start and goal are open
        self.grid[self.start_pos.row][self.start_pos.col] = '.'
        self.grid[self.goal_pos.row][self.goal_pos.col]   = '.'
        # Remove a few random walls to add shortcuts
        for _ in range(6):
            r, c = random.randrange(g), random.randrange(g)
            if (r, c) not in [(self.start_pos.row, self.start_pos.col),
                               (self.goal_pos.row,  self.goal_pos.col)]:
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
        self.current_scan = Position(0, 0)
        self.is_running  = False
        self.iteration   = 0
        self.sweep_count = 0
        self.max_delta   = float('inf')
        self.converged   = False
        self.delta_history = []
        self.show_path   = False
        self.optimal_path = []
        self.agent_pos   = Position(self.start_pos.row, self.start_pos.col)

        # For Policy Iteration: random initial policy
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
        self.reset_values()

    # ── Transition model ──────────────────
    def get_valid_neighbors(self, pos: Position):
        """Returns list of (Position, Direction) for open cells around pos."""
        result = []
        for d in Direction:
            nxt = pos.move(d)
            if nxt.valid() and self.grid[nxt.row][nxt.col] != '#':
                result.append((nxt, d))
        return result

    def transitions(self, pos: Position, action: Direction):
        """
        Returns list of (probability, next_position) given action.
        Deterministic: [(1.0, next)]
        MDP stochastic: 80% forward, 10% each perpendicular (or stay if wall)
        """
        if not self.is_mdp:
            nxt = pos.move(action)
            if not nxt.valid() or self.grid[nxt.row][nxt.col] == '#':
                nxt = pos   # bumps into wall, stays
            return [(1.0, nxt)]

        # Stochastic
        outcomes: Dict[Position, float] = {}

        def add(prob, direction):
            nxt = pos.move(direction)
            if not nxt.valid() or self.grid[nxt.row][nxt.col] == '#':
                nxt = pos  # bounce back
            outcomes[nxt] = outcomes.get(nxt, 0) + prob

        add(Config.MDP_PROB_FORWARD, action)
        for perp_dir in PERP[action]:
            add(Config.MDP_PROB_SIDEWAYS, perp_dir)

        # NOTE: outcomes is {Position: probability}; return consistent (prob, Position)
        return [(prob, nxt) for nxt, prob in outcomes.items()]

    def bellman_value(self, pos: Position, action: Direction) -> float:
        """Q(s, a) = Σ p(s'|s,a) [r + γ V(s')]"""
        total = 0.0
        for prob, nxt in self.transitions(pos, action):
            r = Config.REWARD_GOAL if nxt == self.goal_pos else Config.REWARD_STEP
            total += prob * (r + self.gamma * self.values[nxt])
        return total

    # ── Single-cell update ────────────────
    def update_single_cell(self, pos: Position) -> float:
        """Returns |delta| for convergence tracking."""
        if self.grid[pos.row][pos.col] == '#' or pos == self.goal_pos:
            return 0.0

        neighbors = self.get_valid_neighbors(pos)
        if not neighbors:
            return 0.0

        old_v = self.values[pos]

        if self.use_pi:
            # Policy Evaluation step: evaluate locked policy
            action = self.pi_policy.get(pos, neighbors[0][1])
            new_v  = self.bellman_value(pos, action)
            self.values[pos] = new_v
            # Policy Improvement
            best_v, best_d = max(
                ((self.bellman_value(pos, d), d) for _, d in neighbors),
                key=lambda x: x[0]
            )
            self.pi_policy[pos] = best_d
            self.policy[pos]    = best_d
        else:
            # Value Iteration: greedy maximise
            best_v, best_d = max(
                ((self.bellman_value(pos, d), d) for _, d in neighbors),
                key=lambda x: x[0]
            )
            self.values[pos] = best_v
            self.policy[pos] = best_d

        return abs(self.values[pos] - old_v)

    # ── Scan advance ─────────────────────
    def advance_scan(self):
        """Update current scan cell and move pointer. Returns True if full sweep done."""
        delta = self.update_single_cell(self.current_scan)
        self.iteration += 1

        # Track per-sweep max delta
        if not hasattr(self, '_sweep_delta'):
            self._sweep_delta = 0.0
        self._sweep_delta = max(self._sweep_delta, delta)

        # Advance pointer
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

    # ── Optimal path ─────────────────────
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
        if self.agent_pos == self.goal_pos:
            self.agent_pos = Position(self.start_pos.row, self.start_pos.col)
            return
        if self.agent_pos in self.policy:
            action = self.policy[self.agent_pos]
            outcomes = self.transitions(self.agent_pos, action)
            # Sample from stochastic outcomes
            roll = random.random()
            cumul = 0.0
            chosen = self.agent_pos
            for prob, nxt in outcomes:
                cumul += prob
                if roll <= cumul:
                    chosen = nxt
                    break
            self.agent_pos = chosen

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

    # ── Helpers ───────────────────────────
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

        # Background
        self.screen.fill(Config.BG)

        # ── Title bar ──────────────────────
        pygame.draw.rect(self.screen, Config.PANEL_BG, (0, 0, W, 55))
        self.text("THESEUS  &  THE  MAZE", self.font_xl, Config.TEXT_BRIGHT, 16, 10)

        mode_label = ("Policy Iteration" if state.use_pi else "Value Iteration") + \
                     (" · MDP 80/10/10" if state.is_mdp else " · Deterministic")
        self.text(mode_label, self.font_sm, Config.TEXT_DIM, 16, 40)

        gamma_label = f"γ = {state.gamma:.2f}"
        color  = Config.ACCENT_PI if state.use_pi else Config.ACCENT_VI
        self.pill(gamma_label, W - Config.INFO_W - 90, 16, color)

        # ── Grid area background ───────────
        gx, gy = Config.GRID_OFFSET_X, Config.GRID_OFFSET_Y
        gw = gh = Config.GRID_PX
        pygame.draw.rect(self.screen, Config.GRID_BG, (gx - 4, gy - 4, gw + 8, gh + 8), border_radius=6)

        # Compute value range for normalisation
        vals = [v for p, v in state.values.items()
                if state.grid[p.row][p.col] != '#' and p != state.goal_pos]
        v_min = min(vals) if vals else 0
        v_max = max(vals) if vals else 1
        v_range = max(v_max - v_min, 1e-6)

        # ── Cells ──────────────────────────
        cs = Config.CELL_SIZE
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
                    # Subtle hatch
                    for i in range(0, cs + cs, 16):
                        pygame.draw.line(self.screen, Config.WALL_BORDER,
                                         (cx + i, cy), (cx, cy + i), 1)

                elif pos == state.goal_pos:
                    # Pulsing goal
                    pulse = 0.5 + 0.5 * math.sin(tick * 0.006)
                    color = self.lerp_color(Config.GOAL_B, Config.GOAL_A, pulse)
                    pygame.draw.rect(self.screen, color, rect)

                elif pos == state.current_scan and state.is_running:
                    pygame.draw.rect(self.screen, Config.SCAN_C, rect)

                else:
                    v    = state.values[pos]
                    t    = (v - v_min) / v_range
                    color = self.lerp_color(Config.HEAT_LOW, Config.HEAT_HIGH, t)
                    pygame.draw.rect(self.screen, color, rect)

                # Border
                border_c = (60, 62, 85) if state.grid[r][c] == '#' else (50, 52, 75)
                pygame.draw.rect(self.screen, border_c, rect, 1)

                # Value text (non-wall)
                if state.grid[r][c] != '#':
                    v = state.values[pos]
                    v_color = (255, 255, 255) if pos == state.goal_pos else \
                              (200, 200, 230) if v > (v_min + v_range * 0.5) else (140, 140, 180)
                    if pos == state.goal_pos:
                        self.text(f"+{Config.REWARD_GOAL}", self.font_md, v_color, cx + 5, cy + 5)
                    else:
                        self.text(f"{v:.1f}", self.font_sm, v_color, cx + 5, cy + 5)

                    # Arrow / policy
                    if pos in state.policy and pos != state.goal_pos:
                        self.draw_arrow(cx + cs//2, cy + cs//2,
                                        state.policy[pos], cs // 2 - 8)

        # ── Lookahead lines ────────────────
        if state.is_running:
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

        # ── Info panel ─────────────────────
        self.draw_info_panel(state, tick)

    # ── Arrow helper ──────────────────────
    def draw_arrow(self, cx, cy, d: Direction, size: int):
        dr, dc = d.value
        tip    = (cx + dc * size,     cy + dr * size)
        left   = (cx - dr * size // 3, cy + dc * size // 3)
        right  = (cx + dr * size // 3, cy - dc * size // 3)
        color  = (*Config.ARROW_C[:3], 180)
        pygame.draw.polygon(self.screen, Config.ARROW_C,
                            [tip, left, (cx, cy), right], 0)
        pygame.draw.polygon(self.screen, (*Config.ARROW_C[:3],), [tip, left, right], 1)

    # ── Info panel ────────────────────────
    def draw_info_panel(self, state: MazeState, tick: int):
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
            pygame.draw.line(self.screen, Config.WALL_BORDER,
                             (px, py + line), (px + iw, py + line))
            nl(10)

        # ── Status pill ────────────────────
        if state.converged:
            self.pill(" ✓  CONVERGED ", px, py + line, Config.CONVERGED_C, self.font_md)
            nl(34)
        elif state.is_running:
            pulse = 0.5 + 0.5 * math.sin(tick * 0.012)
            c = self.lerp_color(Config.ACCENT_VI, (180, 220, 255), pulse)
            self.pill(" ▶  RUNNING ", px, py + line, c, self.font_md)
            nl(34)
        else:
            self.pill(" ■  PAUSED ", px, py + line, Config.TEXT_DIM, self.font_md)
            nl(34)

        divider()

        # ── Algorithm params ───────────────
        heading("Algorithm")
        alg  = "Policy Iteration" if state.use_pi else "Value Iteration"
        algo_c = Config.ACCENT_PI if state.use_pi else Config.ACCENT_VI
        row("Method", alg, algo_c)
        row("Environment", "MDP (stoch.)" if state.is_mdp else "Deterministic",
            Config.ACCENT_MDP if state.is_mdp else Config.TEXT_MID)
        row("γ  (discount)", f"{state.gamma:.2f}")
        row("Iterations", state.iteration)
        row("Sweeps", state.sweep_count)
        delta_str = f"{state.max_delta:.4f}" if state.max_delta < 1e9 else "—"
        delta_c = Config.CONVERGED_C if state.max_delta < Config.CONVERGENCE_EPS \
                  else Config.DELTA_BAR_C
        row("Max Δ (sweep)", delta_str, delta_c)
        row("Speed", f"{state.scan_speed} ms/cell")
        nl(4)
        divider()

        # ── Bellman equation display ────────
        heading("Bellman Equation")
        eq_lines = [
            "V(s) = max_a Σ p(s'|s,a)",
            "          · [r + γ V(s')]",
        ]
        if state.is_mdp:
            eq_lines = [
                "V(s) = max_a Σ p(s'|s,a)",
                "         · [r + γ V(s')]",
                "",
                f"p(fwd) = {Config.MDP_PROB_FORWARD:.0%}",
                f"p(side)= {Config.MDP_PROB_SIDEWAYS:.0%} each",
            ]
        for el in eq_lines:
            self.text(el, self.font_tiny, (160, 200, 255), px, py + line)
            nl(16)
        nl(4)
        divider()

        # ── Delta sparkline ────────────────
        heading("Convergence  (max Δ)")
        hist = state.delta_history
        if hist:
            bw   = iw // max(len(hist), 1)
            bw   = max(bw, 2)
            bh   = 40
            bx0  = px
            by0  = py + line
            h_max = max(hist) if max(hist) > 0 else 1
            for i, dv in enumerate(hist):
                hh = int((dv / h_max) * bh)
                bx = bx0 + i * bw
                by = by0 + bh - hh
                t  = dv / (h_max + 1e-6)
                bc = self.lerp_color(Config.CONVERGED_C, Config.DELTA_BAR_C, t)
                pygame.draw.rect(self.screen, bc, (bx, by, bw - 1, hh))
            pygame.draw.rect(self.screen, Config.WALL_BORDER, (bx0, by0, iw, bh), 1)
            nl(bh + 8)
        else:
            self.text("(no data yet)", self.font_sm, Config.TEXT_DIM, px, py + line)
            nl(24)
        divider()

        # ── Controls ──────────────────────
        heading("Controls")
        controls = [
            ("SPACE",      "Start / Pause scan"),
            ("S",          "Step agent"),
            ("R",          "Reset values"),
            ("N",          "New random maze"),
            ("M",          "Toggle MDP mode"),
            ("P",          "Toggle PI / VI"),
            ("G",          "Cycle gamma"),
            ("+  /  -",    "Speed up / down"),
            ("LClick",     "Toggle wall"),
            ("RClick",     "Move goal"),
            ("MClick",     "Move start"),
        ]
        for key, desc in controls:
            key_surf = self.font_tiny.render(key, True, (255, 220, 100))
            self.screen.blit(key_surf, (px, py + line))
            self.text(desc, self.font_tiny, Config.TEXT_DIM, px + 56, py + line)
            nl(17)


# ─────────────────────────────────────────
#  App
# ─────────────────────────────────────────
class App:
    def __init__(self):
        self.screen   = pygame.display.set_mode(
            (Config.WIN_W, Config.WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption("Theseus & the Maze — DP/MDP Workshop")
        self.state    = MazeState()
        self.renderer = Renderer(self.screen)
        self.clock    = pygame.time.Clock()
        self.tick     = 0

    def cell_from_mouse(self, mx, my) -> Optional[Position]:
        c = (mx - Config.GRID_OFFSET_X) // Config.CELL_SIZE
        r = (my - Config.GRID_OFFSET_Y) // Config.CELL_SIZE
        if 0 <= r < Config.GRID_SIZE and 0 <= c < Config.GRID_SIZE:
            return Position(r, c)
        return None

    def run(self):
        while True:
            now = pygame.time.get_ticks()

            # ── Auto-scan tick ─────────────
            if self.state.is_running and \
               now - self.state.last_update_ms >= self.state.scan_speed:
                self.state.advance_scan()
                self.state.last_update_ms = now

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
        if key == pygame.K_SPACE:
            if s.converged:
                s.reset_values()
            s.is_running = not s.is_running

        elif key == pygame.K_s:
            s.step_agent()

        elif key == pygame.K_r:
            s.reset_values()

        elif key == pygame.K_n:
            s.generate_random_maze()

        elif key == pygame.K_m:
            s.is_mdp = not s.is_mdp
            s.reset_values()

        elif key == pygame.K_p:
            s.use_pi = not s.use_pi
            s.reset_values()

        elif key == pygame.K_g:
            s.cycle_gamma()

        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            s.scan_speed = max(Config.SPEED_MIN, s.scan_speed - Config.SPEED_STEP)

        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            s.scan_speed = min(Config.SPEED_MAX, s.scan_speed + Config.SPEED_STEP)


# ─────────────────────────────────────────
if __name__ == "__main__":
    App().run()