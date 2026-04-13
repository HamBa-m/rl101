import pygame
import random
import sys
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from collections import deque

# Initialize Pygame
pygame.init()

# ============================================================================
# CONFIGURATION - All magic numbers centralized
# ============================================================================
class Config:
    # Window settings
    GRID_SIZE = 7
    CELL_SIZE = 100
    WINDOW_SIZE = GRID_SIZE * CELL_SIZE + 20  # 10px padding on each side
    INFO_PANEL_WIDTH = 250
    TOTAL_WIDTH = WINDOW_SIZE + INFO_PANEL_WIDTH
    TOTAL_HEIGHT = WINDOW_SIZE + 50  # Extra for title
    
    # Grid offset
    GRID_OFFSET_X = 10
    GRID_OFFSET_Y = 50
    
    # Animation
    AUTO_STEP_DELAY_MS = 50
    RELAY_FLASH_DURATION_MS = 800  # Increased for better visibility during exploration
    LEARNING_STEP_DELAY_MS = 400  # Delay between learning each relay
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 100, 255)
    LIGHT_BLUE = (100, 200, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    ORANGE = (255, 165, 0)
    YELLOW = (255, 255, 0)
    PURPLE = (128, 0, 128)
    GRAY = (200, 200, 200)
    DARK_GRAY = (100, 100, 100)
    BROWN = (139, 69, 19)
    RELAY_BG = (255, 240, 200)
    RELAY_ACTIVE_BG = (255, 200, 100)  # Brighter when just activated
    GOLD = (255, 215, 0)  # For learning highlight


# ============================================================================
# DATA MODELS
# ============================================================================
class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    
    @property
    def vector(self) -> Tuple[int, int]:
        return self.value
    
    @classmethod
    def from_vector(cls, dr: int, dc: int) -> Optional['Direction']:
        for direction in cls:
            if direction.value == (dr, dc):
                return direction
        return None


@dataclass
class Position:
    row: int
    col: int
    
    def __hash__(self):
        return hash((self.row, self.col))
    
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col
    
    def move(self, direction: Direction) -> 'Position':
        dr, dc = direction.vector
        return Position(self.row + dr, self.col + dc)


class RelayState(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    JUST_ACTIVATED = "just_activated"  # For visual feedback
    JUST_USED = "just_used"  # When relay is consulted


class Relay:
    """Represents a single relay in Shannon's machine"""
    def __init__(self, position: Position):
        self.position = position
        self.remembered_direction: Optional[Direction] = None
        self.activation_count: int = 0
        self.last_activated_time: int = 0
        self.last_used_time: int = 0
        self.state = RelayState.INACTIVE
        
    def activate(self, direction: Direction):
        """Learn and remember a direction"""
        self.remembered_direction = direction
        self.activation_count += 1
        self.last_activated_time = pygame.time.get_ticks()
        self.state = RelayState.JUST_ACTIVATED
        
    def use(self):
        """Mark relay as being consulted"""
        self.last_used_time = pygame.time.get_ticks()
        self.state = RelayState.JUST_USED
        
    def deactivate(self):
        """Clear the relay memory"""
        self.remembered_direction = None
        self.state = RelayState.INACTIVE
        
    def update_state(self):
        """Update visual state based on time"""
        current_time = pygame.time.get_ticks()
        
        # Flash effect fades after delay
        if self.state == RelayState.JUST_ACTIVATED:
            if current_time - self.last_activated_time > Config.RELAY_FLASH_DURATION_MS:
                self.state = RelayState.ACTIVE if self.remembered_direction else RelayState.INACTIVE
                
        elif self.state == RelayState.JUST_USED:
            if current_time - self.last_used_time > Config.RELAY_FLASH_DURATION_MS:
                self.state = RelayState.ACTIVE if self.remembered_direction else RelayState.INACTIVE
    
    @property
    def is_active(self) -> bool:
        return self.remembered_direction is not None


# ============================================================================
# GAME STATE - Separating data from presentation
# ============================================================================
class MazeState:
    """Pure game state - no rendering logic"""
    def __init__(self):
        self.start_pos = Position(0, 0)
        self.grid, self.goal_pos = self._create_maze()
        self.mouse_pos = self.start_pos
        
        # Initialize relays for all walkable cells
        self.relays: Dict[Position, Relay] = {}
        for row in range(Config.GRID_SIZE):
            for col in range(Config.GRID_SIZE):
                if self.grid[row][col] != '#':
                    pos = Position(row, col)
                    self.relays[pos] = Relay(pos)
        
        # Path tracking
        self.current_path: List[Position] = [self.start_pos]
        self.successful_paths: List[List[Position]] = []
        
        # Statistics
        self.steps_taken = 0
        self.episode = 1
        self.episode_step_counts: List[int] = []
        
        # Flag to control when to use relays
        self.first_goal_reached = False
        
    def _create_maze(self) -> Tuple[List[List[str]], Position]:
        """Create a randomized solvable maze grid (same dimensions each run).

        Start is fixed at (0,0). Goal is randomized each run, but guaranteed reachable.
        """

        size = Config.GRID_SIZE
        start = Position(0, 0)

        # Prefer a goal that is "quite far" in terms of shortest-path distance.
        # On a 7x7 grid, the open-grid corner distance is 12; 9+ feels noticeably far.
        max_open_dist = 2 * (size - 1)
        min_goal_dist = max(6, int(0.75 * max_open_dist))

        # Regenerate until solvable. Keep a sane cap to avoid any theoretical infinite loop.
        max_attempts = 500

        # Choose a wall density that creates interesting mazes but still solvable often.
        # We jitter per run and per attempt to keep variety.
        base_wall_prob = random.uniform(0.22, 0.34)

        for attempt in range(max_attempts):
            wall_prob = max(0.10, min(0.45, base_wall_prob + random.uniform(-0.05, 0.05)))
            grid = [['.' for _ in range(size)] for _ in range(size)]

            # Place walls randomly, but never on start/goal.
            for r in range(size):
                for c in range(size):
                    # Keep start open; goal will be chosen later among reachable cells.
                    if (r, c) == (start.row, start.col):
                        continue
                    if random.random() < wall_prob:
                        grid[r][c] = '#'

            grid[start.row][start.col] = 'S'

            distances = self._distances_from(grid, start)
            # Need at least one reachable cell other than start to place a goal.
            if len(distances) <= 1:
                continue

            # Prefer far reachable cells (excluding start). Use '.' cells only.
            far_candidates = [
                p for p, d in distances.items()
                if p != start and grid[p.row][p.col] == '.' and d >= min_goal_dist
            ]

            if far_candidates:
                goal = random.choice(far_candidates)
                grid[goal.row][goal.col] = 'G'
                return grid, goal

            # If no cells meet the minimum distance, still pick the farthest reachable cell.
            candidates = [p for p in distances.keys() if p != start and grid[p.row][p.col] == '.']
            if not candidates:
                continue

            farthest_dist = max(distances[p] for p in candidates)
            farthest_cells = [p for p in candidates if distances[p] == farthest_dist]
            goal = random.choice(farthest_cells)
            grid[goal.row][goal.col] = 'G'
            return grid, goal

        # Fallback: carve a guaranteed path if random generation somehow keeps failing.
        grid = [['.' for _ in range(size)] for _ in range(size)]
        # Choose a random far goal (not start).
        while True:
            goal = Position(random.randrange(size), random.randrange(size))
            if goal == start:
                continue
            if abs(goal.row - start.row) + abs(goal.col - start.col) >= min_goal_dist:
                break

        r, c = start.row, start.col
        while (r, c) != (goal.row, goal.col):
            # Randomly step toward goal.
            if r < goal.row and c < goal.col:
                if random.random() < 0.5:
                    r += 1
                else:
                    c += 1
            elif r < goal.row:
                r += 1
            elif c < goal.col:
                c += 1
            else:
                # If overshoot can't happen due to logic, but keep safe.
                break
            grid[r][c] = '.'

        # Sprinkle some walls away from the carved path.
        for rr in range(size):
            for cc in range(size):
                if (rr, cc) in ((start.row, start.col), (goal.row, goal.col)):
                    continue
                if random.random() < 0.28:
                    grid[rr][cc] = '#'

        grid[start.row][start.col] = 'S'
        grid[goal.row][goal.col] = 'G'
        return grid, goal

    def _reachable_from(self, grid: List[List[str]], start: Position) -> set[Position]:
        """Return set of walkable positions reachable from start (BFS)."""
        return set(self._distances_from(grid, start).keys())

    def _distances_from(self, grid: List[List[str]], start: Position) -> Dict[Position, int]:
        """Return shortest-path distances from start to all reachable cells (BFS)."""
        size = Config.GRID_SIZE

        def is_open(p: Position) -> bool:
            if not (0 <= p.row < size and 0 <= p.col < size):
                return False
            return grid[p.row][p.col] != '#'

        if not is_open(start):
            return {}

        q = deque([start])
        dist: Dict[Position, int] = {start: 0}

        while q:
            cur = q.popleft()
            for direction in Direction:
                nxt = cur.move(direction)
                if nxt not in dist and is_open(nxt):
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)

        return dist

    def _is_solvable(self, grid: List[List[str]], start: Position, goal: Position) -> bool:
        """Compatibility helper: can we get from start to goal?"""
        return goal in self._reachable_from(grid, start)
    
    def get_cell(self, pos: Position) -> str:
        """Get cell content at position"""
        if 0 <= pos.row < Config.GRID_SIZE and 0 <= pos.col < Config.GRID_SIZE:
            return self.grid[pos.row][pos.col]
        return '#'  # Out of bounds treated as wall
    
    def is_walkable(self, pos: Position) -> bool:
        """Check if position is walkable"""
        return self.get_cell(pos) != '#'
    
    def get_valid_moves(self, pos: Position) -> List[Tuple[Position, Direction]]:
        """Get all valid moves from a position"""
        moves = []
        for direction in Direction:
            new_pos = pos.move(direction)
            if self.is_walkable(new_pos):
                moves.append((new_pos, direction))
        return moves
    
    def is_at_goal(self) -> bool:
        """Check if mouse is at goal"""
        return self.mouse_pos == self.goal_pos
    
    def reset_episode(self, forget_relays: bool = False):
        """Reset for new episode"""
        self.mouse_pos = self.start_pos
        self.current_path = [self.start_pos]
        
        if self.steps_taken > 0:
            self.episode_step_counts.append(self.steps_taken)
        
        self.steps_taken = 0
        self.episode += 1
        
        if forget_relays:
            for relay in self.relays.values():
                relay.deactivate()
            # Also reset the first goal flag
            self.first_goal_reached = False
        
        # Update relay states
        for relay in self.relays.values():
            relay.update_state()


# ============================================================================
# GAME LOGIC - Shannon's relay algorithm
# ============================================================================
class TheseusBrain:
    """Implements Shannon's relay-based learning algorithm"""
    
    @staticmethod
    def choose_move(state: MazeState) -> Tuple[Optional[Position], Optional[Direction], str]:
        """
        Choose next move using Shannon's algorithm:
        - Episode 1: Explore randomly (ignore relays)
        - Episode 2+: Use relays if available, else explore
        
        Returns: (new_position, direction, message)
        """
        current_pos = state.mouse_pos
        valid_moves = state.get_valid_moves(current_pos)
        
        if not valid_moves:
            return None, None, "Stuck! No valid moves."
        
        # Only use relays AFTER first goal has been reached
        if state.first_goal_reached:
            # Check for active relay at current position
            current_relay = state.relays.get(current_pos)
            
            if current_relay and current_relay.is_active:
                # Try to use the relay
                remembered_dir = current_relay.remembered_direction
                if remembered_dir is None:
                    # Shouldn't happen if is_active is True, but keep it safe.
                    current_relay.deactivate()
                    new_pos, direction = random.choice(valid_moves)
                    return new_pos, direction, f"Relay empty, exploring: {direction.name}"
                new_pos = current_pos.move(remembered_dir)
                
                # Check if remembered direction is still valid
                if state.is_walkable(new_pos):
                    current_relay.use()  # Mark as used for visual feedback
                    return new_pos, remembered_dir, f"Using relay: {remembered_dir.name}"
                else:
                    # Relay points to wall - deactivate and explore
                    current_relay.deactivate()
                    new_pos, direction = random.choice(valid_moves)
                    return new_pos, direction, f"Relay invalid, exploring: {direction.name}"
        
        # First episode OR no active relay - explore randomly
        new_pos, direction = random.choice(valid_moves)
        episode_num = state.episode
        return new_pos, direction, f"Ep{episode_num} exploring: {direction.name}"
    
    @staticmethod
    def learn_from_path(state: MazeState):
        """
        Shannon's learning mechanism:
        Backtrack through successful path and activate relays
        """
        path = state.current_path
        
        # Activate relay at each step (except last)
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            # Calculate direction
            dr = next_pos.row - current.row
            dc = next_pos.col - current.col
            direction = Direction.from_vector(dr, dc)
            
            if direction and current in state.relays:
                state.relays[current].activate(direction)
        
        # Store successful path
        state.successful_paths.append(path.copy())


# ============================================================================
# RENDERING - All visual presentation logic
# ============================================================================
class Renderer:
    """Handles all drawing operations"""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        
        # Fonts
        self.font_small = pygame.font.Font(None, 16)
        self.font_normal = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 30)
        self.font_title = pygame.font.Font(None, 40)
    
    def draw_maze(self, state: MazeState, learning_pos: Optional[Position] = None,
                  current_mouse_pos: Optional[Position] = None):
        """Draw the maze grid with relays"""
        for row in range(Config.GRID_SIZE):
            for col in range(Config.GRID_SIZE):
                pos = Position(row, col)
                is_learning_cell = (learning_pos is not None and 
                                   pos == learning_pos)
                is_current_cell = (current_mouse_pos is not None and 
                                  pos == current_mouse_pos)
                self._draw_cell(state, pos, is_learning_cell, is_current_cell)
    
    def _draw_cell(self, state: MazeState, pos: Position, is_learning_cell: bool = False,
                   is_current_cell: bool = False):
        """Draw a single cell"""
        x = pos.col * Config.CELL_SIZE + Config.GRID_OFFSET_X
        y = pos.row * Config.CELL_SIZE + Config.GRID_OFFSET_Y
        
        # Draw border (special highlighting for learning or current cell)
        if is_learning_cell:
            border_width = 4
            border_color = Config.GOLD  # Gold for learning
        elif is_current_cell:
            border_width = 3
            border_color = Config.BLUE  # Blue for current position
        else:
            border_width = 2
            border_color = Config.BLACK
        
        pygame.draw.rect(self.screen, border_color, 
                        (x, y, Config.CELL_SIZE, Config.CELL_SIZE), border_width)
        
        cell = state.get_cell(pos)
        
        if cell == '#':
            # Draw wall
            self._draw_wall(x, y)
        else:
            # Draw walkable cell
            self._draw_walkable_cell(state, pos, x, y, cell)
    
    def _draw_wall(self, x: int, y: int):
        """Draw a wall cell"""
        pygame.draw.rect(self.screen, Config.DARK_GRAY, 
                        (x+2, y+2, Config.CELL_SIZE-4, Config.CELL_SIZE-4))
        
        # Brick pattern
        for i in range(0, Config.CELL_SIZE-4, 10):
            pygame.draw.line(self.screen, Config.BLACK, 
                           (x+2, y+2+i), (x+Config.CELL_SIZE-2, y+2+i), 1)
    
    def _draw_walkable_cell(self, state: MazeState, pos: Position, 
                           x: int, y: int, cell: str):
        """Draw a walkable cell with potential relay"""
        relay = state.relays.get(pos)
        
        # Background color based on relay state
        if relay and relay.is_active:
            if relay.state == RelayState.JUST_ACTIVATED:
                bg_color = Config.RELAY_ACTIVE_BG  # Bright flash when learned
            elif relay.state == RelayState.JUST_USED:
                bg_color = (100, 255, 100)  # Bright green flash when consulted!
            else:
                bg_color = Config.RELAY_BG  # Normal active color
        else:
            bg_color = Config.WHITE
        
        pygame.draw.rect(self.screen, bg_color, 
                        (x+2, y+2, Config.CELL_SIZE-4, Config.CELL_SIZE-4))
        
        # Draw relay if active
        if relay and relay.is_active:
            self._draw_relay(relay, x, y)
        
        # Draw start/goal markers
        if cell == 'S':
            self._draw_start_marker(x, y)
        elif cell == 'G':
            self._draw_goal_marker(x, y)
    
    def _draw_relay(self, relay: Relay, x: int, y: int):
        """Draw relay symbol with direction arrow"""
        center_x = x + Config.CELL_SIZE // 2
        center_y = y + Config.CELL_SIZE // 2
        
        # Draw relay coil
        coil_x = center_x - 15
        coil_y = center_y - 10
        pygame.draw.rect(self.screen, Config.BROWN, (coil_x, coil_y, 30, 20))
        pygame.draw.rect(self.screen, Config.BLACK, (coil_x, coil_y, 30, 20), 2)
        
        # Draw direction arrow
        if relay.remembered_direction:
            arrow_color = Config.ORANGE
            if relay.state == RelayState.JUST_USED:
                arrow_color = Config.GREEN  # Green when just used
            
            self._draw_direction_arrow(center_x, center_y, 
                                      relay.remembered_direction, arrow_color)
        
        # Show activation count
        if relay.activation_count > 0:
            count_text = self.font_small.render(f"×{relay.activation_count}", 
                                               True, Config.BLACK)
            self.screen.blit(count_text, (x+70, y+75))
    
    def _draw_direction_arrow(self, cx: int, cy: int, 
                             direction: Direction, color: tuple):
        """Draw arrow pointing in direction"""
        if direction == Direction.UP:
            points = [(cx, cy-15), (cx-8, cy-5), (cx+8, cy-5)]
        elif direction == Direction.DOWN:
            points = [(cx, cy+15), (cx-8, cy+5), (cx+8, cy+5)]
        elif direction == Direction.LEFT:
            points = [(cx-15, cy), (cx-5, cy-8), (cx-5, cy+8)]
        elif direction == Direction.RIGHT:
            points = [(cx+15, cy), (cx+5, cy-8), (cx+5, cy+8)]
        else:
            return
        
        pygame.draw.polygon(self.screen, color, points)
    
    def _draw_start_marker(self, x: int, y: int):
        """Draw start position marker"""
        center_x = x + Config.CELL_SIZE // 2
        center_y = y + Config.CELL_SIZE // 2
        pygame.draw.circle(self.screen, Config.GREEN, (center_x, center_y), 15)
        text = self.font_small.render("START", True, Config.BLACK)
        self.screen.blit(text, (x+30, y+40))
    
    def _draw_goal_marker(self, x: int, y: int):
        """Draw goal position marker"""
        center_x = x + Config.CELL_SIZE // 2
        center_y = y + Config.CELL_SIZE // 2
        pygame.draw.circle(self.screen, Config.RED, (center_x, center_y), 15)
        text = self.font_small.render("GOAL", True, Config.WHITE)
        self.screen.blit(text, (x+35, y+40))
    
    def draw_path(self, path: List[Position]):
        """Draw the path taken by the mouse"""
        if len(path) < 2:
            return
        
        points = []
        for pos in path:
            x = pos.col * Config.CELL_SIZE + Config.GRID_OFFSET_X + Config.CELL_SIZE // 2
            y = pos.row * Config.CELL_SIZE + Config.GRID_OFFSET_Y + Config.CELL_SIZE // 2
            points.append((x, y))
        
        pygame.draw.lines(self.screen, Config.YELLOW, False, points, 3)
    
    def draw_mouse(self, pos: Position):
        """Draw Theseus the mouse with improved design"""
        x = pos.col * Config.CELL_SIZE + Config.GRID_OFFSET_X + Config.CELL_SIZE // 2
        y = pos.row * Config.CELL_SIZE + Config.GRID_OFFSET_Y + Config.CELL_SIZE // 2
        
        # Body (larger, more prominent)
        pygame.draw.circle(self.screen, Config.BLUE, (x, y), 18)
        
        # Head
        pygame.draw.circle(self.screen, Config.BLUE, (x-8, y-12), 12)
        
        # Ears (more prominent)
        pygame.draw.circle(self.screen, Config.BLUE, (x-18, y-20), 6)
        pygame.draw.circle(self.screen, Config.BLUE, (x+2, y-20), 6)
        pygame.draw.circle(self.screen, (200, 100, 100), (x-18, y-20), 3)  # Inner ear
        pygame.draw.circle(self.screen, (200, 100, 100), (x+2, y-20), 3)
        
        # Eyes
        pygame.draw.circle(self.screen, Config.WHITE, (x-12, y-14), 4)
        pygame.draw.circle(self.screen, Config.BLACK, (x-12, y-14), 2)
        pygame.draw.circle(self.screen, Config.WHITE, (x-2, y-14), 4)
        pygame.draw.circle(self.screen, Config.BLACK, (x-2, y-14), 2)
        
        # Nose
        pygame.draw.circle(self.screen, Config.RED, (x-7, y-6), 2)
        
        # Whiskers
        pygame.draw.line(self.screen, Config.BLACK, (x-7, y-4), (x-18, y-2), 1)
        pygame.draw.line(self.screen, Config.BLACK, (x-7, y), (x-18, y+3), 1)
        pygame.draw.line(self.screen, Config.BLACK, (x-7, y+4), (x-18, y+6), 1)
        
        # Tail (curved appearance)
        pygame.draw.line(self.screen, Config.BLUE, (x+18, y), (x+28, y-8), 4)
        pygame.draw.line(self.screen, Config.BLUE, (x+28, y-8), (x+32, y-2), 3)
        
        # Label
        text = self.font_small.render("THESEUS", True, Config.BLACK)
        text_rect = text.get_rect(center=(x, y+32))
        pygame.draw.rect(self.screen, Config.BLUE, text_rect.inflate(8, 4))
        self.screen.blit(text, text_rect)
    
    def draw_title(self):
        """Draw main title"""
        title = self.font_title.render("Claude Shannon's Theseus Machine (1950)", 
                                      True, Config.BLACK)
        self.screen.blit(title, (150, 10))
        
        subtitle = self.font_normal.render("A Mechanical Mouse with Relay Memory", 
                                          True, Config.DARK_GRAY)
        self.screen.blit(subtitle, (250, 35))
    
    def draw_info_panel(self, state: MazeState, message: str):
        """Draw information panel"""
        panel_x = Config.WINDOW_SIZE + 20
        
        # Title
        title = self.font_large.render("RELAY STATUS", True, Config.BLACK)
        self.screen.blit(title, (panel_x, 60))
        
        # Statistics
        active_relays = sum(1 for r in state.relays.values() if r.is_active)
        total_activations = sum(r.activation_count for r in state.relays.values())
        
        stats = [
            f"Episode: {state.episode}",
            f"Steps: {state.steps_taken}",
            f"Active Relays: {active_relays}/{len(state.relays)}",
            f"Total Memories: {total_activations}",
            f"At Goal: {'Yes' if state.is_at_goal() else 'No'}"
        ]
        
        y = 110
        for stat in stats:
            text = self.font_normal.render(stat, True, Config.BLACK)
            self.screen.blit(text, (panel_x, y))
            y += 25
        
        # Legend
        y += 20
        legend_title = self.font_normal.render("Relay States:", True, Config.BLACK)
        self.screen.blit(legend_title, (panel_x, y))
        y += 30
        
        # Just activated (orange) - NOW THIS APPEARS AS MOUSE MOVES!
        pygame.draw.rect(self.screen, Config.RELAY_ACTIVE_BG, (panel_x, y, 30, 20))
        pygame.draw.rect(self.screen, Config.BLACK, (panel_x, y, 30, 20), 1)
        text = self.font_small.render("= Just Created!", True, Config.BLACK)
        self.screen.blit(text, (panel_x+35, y+2))
        y += 30
        
        # Active relay (beige)
        pygame.draw.rect(self.screen, Config.RELAY_BG, (panel_x, y, 30, 20))
        pygame.draw.rect(self.screen, Config.BLACK, (panel_x, y, 30, 20), 1)
        text = self.font_small.render("= Has Memory", True, Config.BLACK)
        self.screen.blit(text, (panel_x+35, y+2))
        y += 30
        
        # # Just used (BRIGHT green)
        # pygame.draw.rect(self.screen, (100, 255, 100), (panel_x, y, 30, 20))
        # pygame.draw.rect(self.screen, Config.BLACK, (panel_x, y, 30, 20), 1)
        # text = self.font_small.render("= Being Used!", True, Config.BLACK)
        # self.screen.blit(text, (panel_x+35, y+2))
        # y += 30
        
        # Inactive
        pygame.draw.rect(self.screen, Config.WHITE, (panel_x, y, 30, 20))
        pygame.draw.rect(self.screen, Config.BLACK, (panel_x, y, 30, 20), 1)
        text = self.font_small.render("= No Memory", True, Config.BLACK)
        self.screen.blit(text, (panel_x+35, y+2))
        y += 30
        
        # Arrow example
        pygame.draw.polygon(self.screen, Config.ORANGE, 
                          [(panel_x+15, y+5), (panel_x+7, y+15), (panel_x+23, y+15)])
        text = self.font_small.render("= Remembered Dir", True, Config.BLACK)
        self.screen.blit(text, (panel_x+35, y+7))
        
        # Message box
        y += 50
        self._draw_message_box(panel_x, y, message)
    
    def _draw_message_box(self, x: int, y: int, message: str):
        """Draw message box with word wrapping"""
        box_width = Config.INFO_PANEL_WIDTH - 40
        box_height = 80
        
        pygame.draw.rect(self.screen, Config.LIGHT_BLUE, 
                        (x, y, box_width, box_height))
        pygame.draw.rect(self.screen, Config.BLACK, 
                        (x, y, box_width, box_height), 2)
        
        # Word wrap
        words = message.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if self.font_normal.size(test_line)[0] <= box_width - 20:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Draw lines (max 3)
        for i, line in enumerate(lines[:3]):
            text = self.font_normal.render(line, True, Config.BLACK)
            self.screen.blit(text, (x+10, y+10 + i*22))


# ============================================================================
# UI COMPONENTS
# ============================================================================
class Button:
    def __init__(self, x: int, y: int, width: int, height: int, 
                 text: str, color: tuple, action):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.action = action
        self.font = pygame.font.Font(None, 22)
        self.hovered = False
    
    def draw(self, screen: pygame.Surface):
        # Lighter color on hover
        color = self.color
        if self.hovered:
            color = tuple(min(c + 30, 255) for c in self.color)
        
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, Config.BLACK, self.rect, 2)
        
        text_surface = self.font.render(self.text, True, Config.WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.action()
                return True
        return False


# ============================================================================
# MAIN GAME CONTROLLER
# ============================================================================
class TheseusGame:
    """Main game controller - coordinates everything"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((Config.TOTAL_WIDTH, Config.TOTAL_HEIGHT))
        pygame.display.set_caption("Shannon's Theseus Machine, Virtualized (1950)")
        
        self.state = MazeState()
        self.renderer = Renderer(self.screen)
        self.brain = TheseusBrain()
        
        self.message = "Episode 1: Random exploration. Relays appear as you move!"
        self.running = True
        self.auto_running = False
        self.last_step_time = 0
        
        # Display options
        self.show_path = False  # Path visualization disabled by default
        
        self._create_buttons()
    
    def _create_buttons(self):
        """Create UI buttons"""
        button_width = 180
        button_height = 40
        x = Config.WINDOW_SIZE + 30
        y_start = 520
        spacing = 50
        
        self.buttons = [
            Button(x, y_start, button_width, button_height, 
                  "STEP", Config.GREEN, self.step),
            Button(x, y_start + spacing, button_width, button_height, 
                  "AUTO RUN", Config.BLUE, self.toggle_auto_run),
            Button(x, y_start + spacing*2, button_width, button_height, 
                  "RESET", Config.ORANGE, self.reset),
            Button(x, y_start + spacing*3, button_width, button_height, 
                  "RESET & FORGET", Config.RED, self.reset_and_forget),
            Button(x, y_start + spacing*4, button_width, button_height, 
                  "QUIT", Config.BLACK, self.quit_game)
        ]
    
    def step(self):
        """Execute one step of the algorithm"""
        if self.state.is_at_goal():
            self.message = "Already at goal! Click RESET to try again."
            return
        
        # Remember current position before moving
        old_pos = self.state.mouse_pos
        
        # Choose move using Shannon's algorithm
        new_pos, direction, msg = self.brain.choose_move(self.state)
        
        if new_pos is None:
            self.message = msg
            return
        
        # Execute move
        self.state.mouse_pos = new_pos
        self.state.current_path.append(new_pos)
        self.state.steps_taken += 1
        self.message = msg
        
        # ⚡ IMMEDIATE LEARNING: Activate relay at the cell we just left!
        if direction and old_pos in self.state.relays:
            self.state.relays[old_pos].activate(direction)
        
        # Check if goal reached
        if self.state.is_at_goal():
            # Mark that first goal has been reached (enables relay usage)
            if not self.state.first_goal_reached:
                self.state.first_goal_reached = True
                self.message = "FIRST GOAL! Relays will be used from next episode."
            else:
                self.message = "GOAL REACHED!"
            
            self.state.successful_paths.append(self.state.current_path.copy())
            self.auto_running = False
    
    def toggle_auto_run(self):
        """Toggle automatic stepping"""
        self.auto_running = not self.auto_running
        self.last_step_time = pygame.time.get_ticks()
    
    def reset(self):
        """Reset episode, keep memories"""
        active_count = sum(1 for r in self.state.relays.values() if r.is_active)
        self.state.reset_episode(forget_relays=False)
        self.message = f"Episode {self.state.episode}: {active_count} active relays"
        self.auto_running = False
    
    def reset_and_forget(self):
        """Reset and clear all memories"""
        self.state.reset_episode(forget_relays=True)
        self.message = f"Episode {self.state.episode}: All memories cleared!"
        self.auto_running = False
    
    def quit_game(self):
        """Exit the game"""
        self.running = False
    
    def handle_events(self):
        """Process all events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.step()
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_f:
                    self.reset_and_forget()
                elif event.key == pygame.K_a:
                    self.toggle_auto_run()
            
            # Handle button events
            for button in self.buttons:
                button.handle_event(event)
    
    def update(self):
        """Update game state"""
        current_time = pygame.time.get_ticks()
        
        # Auto-run logic
        if self.auto_running and not self.state.is_at_goal():
            if current_time - self.last_step_time > Config.AUTO_STEP_DELAY_MS:
                self.step()
                self.last_step_time = current_time
        
        # Update relay visual states
        for relay in self.state.relays.values():
            relay.update_state()
    
    def render(self):
        """Render everything"""
        self.screen.fill(Config.WHITE)
        
        self.renderer.draw_title()
        self.renderer.draw_maze(self.state, None, self.state.mouse_pos)
        
        # Only draw path if enabled
        if self.show_path:
            self.renderer.draw_path(self.state.current_path)
        
        self.renderer.draw_mouse(self.state.mouse_pos)
        self.renderer.draw_info_panel(self.state, self.message)
        
        for button in self.buttons:
            button.draw(self.screen)
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    game = TheseusGame()
    game.run()