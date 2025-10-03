use piston_window::*;
use std::collections::{LinkedList, VecDeque, HashMap, BinaryHeap};
use std::cmp::Ordering;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

const WINDOW_WIDTH: f64 = 800.0;
const WINDOW_HEIGHT: f64 = 600.0;
const GRID_SIZE: f64 = 20.0;
const GRID_WIDTH: u32 = (600.0 / GRID_SIZE) as u32; // Game area is 600x600
const GRID_HEIGHT: u32 = (600.0 / GRID_SIZE) as u32;
const STATS_PANEL_WIDTH: f64 = 200.0; // Right panel for stats

// RL Constants
const STATE_SIZE: usize = 16; // Improved state representation
const ACTION_SIZE: usize = 4; // Up, Down, Left, Right
const MEMORY_SIZE: usize = 50000; // Increased memory
const BATCH_SIZE: usize = 64; // Larger batch size
const LEARNING_RATE: f64 = 0.0005; // Lower learning rate for stability
const GAMMA: f64 = 0.99; // Higher discount factor
const EPSILON_START: f64 = 1.0;
const EPSILON_END: f64 = 0.01;
const EPSILON_DECAY: f64 = 0.9995; // Slower decay
const TARGET_UPDATE_FREQ: usize = 500; // Update target network more frequently

#[derive(Clone, PartialEq, Debug, Copy)]
enum Direction {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl Direction {
    fn from_action(action: usize) -> Self {
        match action {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => Direction::Up,
        }
    }
}

// A* Pathfinding Implementation
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Node {
    x: u32,
    y: u32,
}

impl Node {
    fn new(x: u32, y: u32) -> Self {
        Node { x, y }
    }
    
    fn manhattan_distance(&self, other: &Node) -> u32 {
        ((self.x as i32 - other.x as i32).abs() + (self.y as i32 - other.y as i32).abs()) as u32
    }
    
    fn get_neighbors(&self) -> Vec<Node> {
        let mut neighbors = Vec::new();
        let directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]; // Up, Down, Left, Right
        
        for (dx, dy) in directions.iter() {
            let new_x = self.x as i32 + dx;
            let new_y = self.y as i32 + dy;
            
            if new_x >= 0 && new_x < GRID_WIDTH as i32 && new_y >= 0 && new_y < GRID_HEIGHT as i32 {
                neighbors.push(Node::new(new_x as u32, new_y as u32));
            }
        }
        neighbors
    }
}

#[derive(Clone, PartialEq, Eq)]
struct AStarNode {
    node: Node,
    g_cost: u32,    // Cost from start
    h_cost: u32,    // Heuristic cost to goal
    f_cost: u32,    // Total cost (g + h)
    parent: Option<Node>,
}

impl AStarNode {
    fn new(node: Node, g_cost: u32, h_cost: u32, parent: Option<Node>) -> Self {
        AStarNode {
            node,
            g_cost,
            h_cost,
            f_cost: g_cost + h_cost,
            parent,
        }
    }
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (BinaryHeap is max-heap by default)
        other.f_cost.cmp(&self.f_cost)
            .then_with(|| other.h_cost.cmp(&self.h_cost))
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct AStar {
    obstacles: Vec<Node>,
}

impl AStar {
    fn new() -> Self {
        AStar {
            obstacles: Vec::new(),
        }
    }
    
    fn set_obstacles(&mut self, obstacles: Vec<Node>) {
        self.obstacles = obstacles;
    }
    
    fn is_obstacle(&self, node: &Node) -> bool {
        self.obstacles.contains(node)
    }
    
    fn find_path(&self, start: Node, goal: Node) -> Option<Vec<Node>> {
        if start == goal {
            return Some(vec![start]);
        }
        
        let mut open_set = BinaryHeap::new();
        let mut closed_set = HashMap::new();
        let mut g_scores = HashMap::new();
        
        // Initialize start node
        let start_h = start.manhattan_distance(&goal);
        open_set.push(AStarNode::new(start, 0, start_h, None));
        g_scores.insert(start, 0);
        
        while let Some(current) = open_set.pop() {
            if current.node == goal {
                // Reconstruct path
                return Some(self.reconstruct_path(&closed_set, current.node));
            }
            
            closed_set.insert(current.node, current.parent);
            
            for neighbor in current.node.get_neighbors() {
                if self.is_obstacle(&neighbor) || closed_set.contains_key(&neighbor) {
                    continue;
                }
                
                let tentative_g = current.g_cost + 1;
                
                if let Some(&existing_g) = g_scores.get(&neighbor) {
                    if tentative_g >= existing_g {
                        continue;
                    }
                }
                
                g_scores.insert(neighbor, tentative_g);
                let h_cost = neighbor.manhattan_distance(&goal);
                open_set.push(AStarNode::new(neighbor, tentative_g, h_cost, Some(current.node)));
            }
        }
        
        None // No path found
    }
    
    fn reconstruct_path(&self, came_from: &HashMap<Node, Option<Node>>, mut current: Node) -> Vec<Node> {
        let mut path = vec![current];
        
        while let Some(&Some(parent)) = came_from.get(&current) {
            current = parent;
            path.push(current);
        }
        
        path.reverse();
        path
    }
    
    fn get_next_direction(&self, start: Node, goal: Node) -> Option<Direction> {
        if let Some(path) = self.find_path(start, goal) {
            if path.len() >= 2 {
                let next = path[1];
                let current = path[0];
                
                if next.x > current.x { return Some(Direction::Right); }
                if next.x < current.x { return Some(Direction::Left); }
                if next.y > current.y { return Some(Direction::Down); }
                if next.y < current.y { return Some(Direction::Up); }
            }
        }
        None
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct GameSession {
    hits: u32,
    misses: u32,
    total_moves: u32,
    max_score: u32,
    games_played: u32,
    timestamp: u64,
}

impl GameSession {
    fn new() -> Self {
        GameSession {
            hits: 0,
            misses: 0,
            total_moves: 0,
            max_score: 0,
            games_played: 0,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }

    fn hit_rate(&self) -> f64 {
        if self.total_moves == 0 {
            0.0
        } else {
            (self.hits as f64 / self.total_moves as f64) * 100.0
        }
    }

    // fn miss_rate(&self) -> f64 {
    //     if self.total_moves == 0 {
    //         0.0
    //     } else {
    //         (self.misses as f64 / self.total_moves as f64) * 100.0
    //     }
    // }
}

#[derive(Clone, Serialize, Deserialize)]
struct PointsTable {
    current_session: GameSession,
    all_time_best: GameSession,
    sessions: Vec<GameSession>,
}

impl PointsTable {
    fn new() -> Self {
        PointsTable {
            current_session: GameSession::new(),
            all_time_best: GameSession::new(),
            sessions: Vec::new(),
        }
    }

    fn record_hit(&mut self) {
        self.current_session.hits += 1;
        self.current_session.total_moves += 1;
    }

    fn record_miss(&mut self) {
        self.current_session.misses += 1;
        self.current_session.total_moves += 1;
    }

    fn end_game(&mut self, final_score: u32) {
        self.current_session.games_played += 1;
        if final_score > self.current_session.max_score {
            self.current_session.max_score = final_score;
        }

        // Check if this session beats all-time best
        if self.current_session.max_score > self.all_time_best.max_score ||
           (self.current_session.max_score == self.all_time_best.max_score && 
            self.current_session.hit_rate() > self.all_time_best.hit_rate()) {
            self.all_time_best = self.current_session.clone();
        }
    }

    // fn new_session(&mut self) {
    //     if self.current_session.total_moves > 0 {
    //         self.sessions.push(self.current_session.clone());
    //     }
    //     self.current_session = GameSession::new();
    // }

    fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(filename, json)?;
        Ok(())
    }

    fn load(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(filename)?;
        let table: PointsTable = serde_json::from_str(&json)?;
        Ok(table)
    }
}

#[derive(Clone)]
struct Position {
    x: u32,
    y: u32,
}

struct Snake {
    body: LinkedList<Position>,
    direction: Direction,
    next_direction: Direction,
}

impl Snake {
    fn new() -> Self {
        let mut body = LinkedList::new();
        body.push_back(Position { x: 10, y: 10 });
        body.push_back(Position { x: 9, y: 10 });
        body.push_back(Position { x: 8, y: 10 });

        Snake {
            body,
            direction: Direction::Right,
            next_direction: Direction::Right,
        }
    }

    fn update_direction(&mut self, new_direction: Direction) {
        match (&self.direction, &new_direction) {
            (Direction::Up, Direction::Down) => return,
            (Direction::Down, Direction::Up) => return,
            (Direction::Left, Direction::Right) => return,
            (Direction::Right, Direction::Left) => return,
            _ => self.next_direction = new_direction,
        }
    }

    fn move_forward(&mut self) -> Option<Position> {
        self.direction = self.next_direction.clone();

        let head = self.body.front().unwrap().clone();
        
        // Calculate new position using signed integers to avoid overflow
        let (new_x, new_y) = match self.direction {
            Direction::Up => (head.x as i32, head.y as i32 - 1),
            Direction::Down => (head.x as i32, head.y as i32 + 1),
            Direction::Left => (head.x as i32 - 1, head.y as i32),
            Direction::Right => (head.x as i32 + 1, head.y as i32),
        };

        // Convert back to unsigned, using max values to indicate out-of-bounds
        let new_head = Position {
            x: if new_x < 0 || new_x >= GRID_WIDTH as i32 { u32::MAX } else { new_x as u32 },
            y: if new_y < 0 || new_y >= GRID_HEIGHT as i32 { u32::MAX } else { new_y as u32 },
        };

        self.body.push_front(new_head.clone());
        Some(self.body.pop_back().unwrap())
    }

    fn check_boundary_collision(&self) -> bool {
        let head = self.body.front().unwrap();
        head.x >= GRID_WIDTH || head.y >= GRID_HEIGHT
    }

    fn grow(&mut self) {
        let tail = self.body.back().unwrap().clone();
        self.body.push_back(tail);
    }

    fn check_collision(&self) -> bool {
        let head = self.body.front().unwrap();
        for segment in self.body.iter().skip(1) {
            if head.x == segment.x && head.y == segment.y {
                return true;
            }
        }
        false
    }

    fn get_head(&self) -> &Position {
        self.body.front().unwrap()
    }
}

struct Food {
    position: Position,
}

impl Food {
    fn new() -> Self {
        Food {
            position: Position {
                x: (rand::random::<u32>() % GRID_WIDTH),
                y: (rand::random::<u32>() % GRID_HEIGHT),
            },
        }
    }

    fn respawn(&mut self, snake: &Snake) {
        loop {
            let new_pos = Position {
                x: (rand::random::<u32>() % GRID_WIDTH),
                y: (rand::random::<u32>() % GRID_HEIGHT),
            };
            
            // Make sure food doesn't spawn on snake
            let mut valid = true;
            for segment in &snake.body {
                if segment.x == new_pos.x && segment.y == new_pos.y {
                    valid = false;
                    break;
                }
            }
            
            if valid {
                self.position = new_pos;
                break;
            }
        }
    }
}

struct Game {
    snake: Snake,
    food: Food,
    score: u32,
    game_over: bool,
    steps_without_food: u32,
    points_table: PointsTable,
    pathfinder: AStar,
}

impl Game {
    fn new() -> Self {
        let snake = Snake::new();
        let mut food = Food::new();
        
        // Ensure food doesn't spawn on snake initially
        food.respawn(&snake);
        
        // Try to load existing points table, or create new one
        let points_table = PointsTable::load("points_table.json").unwrap_or_else(|_| PointsTable::new());
        
        Game {
            snake,
            food,
            score: 0,
            game_over: false,
            steps_without_food: 0,
            points_table,
            pathfinder: AStar::new(),
        }
    }

    fn reset(&mut self) {
        *self = Game::new();
    }

    fn update(&mut self) -> f64 {
        if self.game_over {
            return -100.0; // Large negative reward for game over
        }

        let old_distance = self.distance_to_food();
        let old_manhattan = self.manhattan_distance_to_food();
        
        if let Some(_) = self.snake.move_forward() {
            let head = self.snake.body.front().unwrap();
            
            // Check boundary collision first
            if self.snake.check_boundary_collision() {
                self.game_over = true;
                self.points_table.end_game(self.score);
                // Save points table
                let _ = self.points_table.save("points_table.json");
                return -1000.0; // MASSIVE negative reward for hitting boundary
            }
            
            // Check if snake ate food
            if head.x == self.food.position.x && head.y == self.food.position.y {
                self.snake.grow();
                self.score += 10;
                self.food.respawn(&self.snake);
                self.steps_without_food = 0;
                self.points_table.record_hit(); // Record hit
                return 500.0; // Very large positive reward for eating food
            }

            // Check collision with self
            if self.snake.check_collision() {
                self.game_over = true;
                self.points_table.end_game(self.score);
                // Save points table
                let _ = self.points_table.save("points_table.json");
                return -100.0; // Large negative reward for collision
            }

            self.steps_without_food += 1;
            
            // Penalty for taking too long without eating food
            if self.steps_without_food > 200 {
                self.game_over = true;
                self.points_table.end_game(self.score);
                // Save points table
                let _ = self.points_table.save("points_table.json");
                return -100.0;
            }

            let new_distance = self.distance_to_food();
            let new_manhattan = self.manhattan_distance_to_food();
            
            // Only record miss if we're moving away from food significantly
            if new_distance > old_distance + 0.5 {
                self.points_table.record_miss();
            }
            
            // Enhanced reward system for optimal pathfinding
            // 1. Manhattan distance reward (better for grid-based movement)
            let manhattan_reward = if new_manhattan < old_manhattan {
                75.0 // Large reward for getting closer via Manhattan distance
            } else if new_manhattan > old_manhattan {
                -15.0 // Penalty for getting further
            } else {
                0.0 // Neutral for same distance
            };
            
            // 2. Euclidean distance reward (secondary)
            let euclidean_reward = if new_distance < old_distance {
                25.0 // Smaller reward for Euclidean improvement
            } else if new_distance > old_distance {
                -5.0 // Small penalty for getting further
            } else {
                0.0
            };
            
            // 3. Directional alignment bonus - reward moving in optimal direction
            let optimal_direction = self.get_optimal_direction();
            let direction_bonus = if let Some(optimal_dir) = optimal_direction {
                if self.snake.direction == optimal_dir {
                    30.0 // Bonus for moving in optimal direction
                } else {
                    // Check if we're moving perpendicular (zigzag)
                    let is_perpendicular = match (optimal_dir, self.snake.direction) {
                        (Direction::Up | Direction::Down, Direction::Left | Direction::Right) => true,
                        (Direction::Left | Direction::Right, Direction::Up | Direction::Down) => true,
                        _ => false,
                    };
                    if is_perpendicular {
                        -10.0 // Penalty for zigzag movement
                    } else {
                        0.0
                    }
                }
            } else {
                0.0
            };
            
            // 4. Wall proximity penalty - discourage getting too close to walls
            let head = self.snake.body.front().unwrap();
            let wall_penalty = if head.x <= 1 || head.x >= GRID_WIDTH - 2 || 
                                  head.y <= 1 || head.y >= GRID_HEIGHT - 2 {
                -25.0 // Penalty for being very close to walls
            } else if head.x <= 2 || head.x >= GRID_WIDTH - 3 || 
                      head.y <= 2 || head.y >= GRID_HEIGHT - 3 {
                -5.0 // Small penalty for being close to walls
            } else {
                0.0 // No penalty for being away from walls
            };

            // 5. Small reward for staying alive
            let survival_reward = 1.0;
            
            manhattan_reward + euclidean_reward + direction_bonus + wall_penalty + survival_reward
        } else {
            -100.0 // Shouldn't happen, but handle gracefully
        }
    }

    fn distance_to_food(&self) -> f64 {
        let head = self.snake.get_head();
        let dx = (head.x as i32 - self.food.position.x as i32).abs() as f64;
        let dy = (head.y as i32 - self.food.position.y as i32).abs() as f64;
        (dx * dx + dy * dy).sqrt()
    }

    fn manhattan_distance_to_food(&self) -> f64 {
        let head = self.snake.get_head();
        let dx = (head.x as i32 - self.food.position.x as i32).abs() as f64;
        let dy = (head.y as i32 - self.food.position.y as i32).abs() as f64;
        dx + dy
    }

    fn get_optimal_direction(&self) -> Option<Direction> {
        let head = self.snake.get_head();
        let food = &self.food.position;
        
        let dx = food.x as i32 - head.x as i32;
        let dy = food.y as i32 - head.y as i32;
        
        // Prioritize the axis with larger distance
        if dx.abs() > dy.abs() {
            if dx > 0 { Some(Direction::Right) } else { Some(Direction::Left) }
        } else if dy.abs() > dx.abs() {
            if dy > 0 { Some(Direction::Down) } else { Some(Direction::Up) }
        } else if dx != 0 {
            // Equal distances, prefer horizontal movement
            if dx > 0 { Some(Direction::Right) } else { Some(Direction::Left) }
        } else if dy != 0 {
            if dy > 0 { Some(Direction::Down) } else { Some(Direction::Up) }
        } else {
            None // Already at food
        }
    }

    fn get_astar_direction(&mut self) -> Option<Direction> {
        let head = self.snake.get_head();
        let food = &self.food.position;
        
        // Convert positions to Node format
        let start = Node::new(head.x, head.y);
        let goal = Node::new(food.x, food.y);
        
        // Set obstacles (snake body excluding head)
        let obstacles: Vec<Node> = self.snake.body.iter()
            .skip(1) // Skip head
            .map(|pos| Node::new(pos.x, pos.y))
            .collect();
        
        self.pathfinder.set_obstacles(obstacles);
        
        // Get next direction from A*
        self.pathfinder.get_next_direction(start, goal)
    }

    fn get_intelligent_direction(&mut self) -> Option<Direction> {
        let snake_length = self.snake.body.len();
        
        // Use A* when snake is longer (more obstacles) or in tight spaces
        let should_use_astar = snake_length > 5 || self.is_in_tight_space();
        
        if should_use_astar {
            // Try A* first for complex scenarios
            if let Some(astar_dir) = self.get_astar_direction() {
                if !self.is_danger_ahead(astar_dir) {
                    return Some(astar_dir);
                }
            }
        }
        
        // Fallback to simple optimal direction for simple scenarios or when A* fails
        self.get_hybrid_direction()
    }
    
    fn is_in_tight_space(&self) -> bool {
        let mut blocked_directions = 0;
        
        // Check how many directions are blocked
        for direction in [Direction::Up, Direction::Down, Direction::Left, Direction::Right].iter() {
            if self.is_danger_ahead(*direction) {
                blocked_directions += 1;
            }
        }
        
        // Consider it tight if 2 or more directions are blocked
        blocked_directions >= 2
    }

    fn get_hybrid_direction(&mut self) -> Option<Direction> {
        // Try A* first
        if let Some(astar_dir) = self.get_astar_direction() {
            // Check if A* direction is safe (not into immediate danger)
            if !self.is_danger_ahead(astar_dir) {
                return Some(astar_dir);
            }
        }
        
        // Fallback to simple optimal direction if A* fails or is dangerous
        let simple_dir = self.get_optimal_direction();
        if let Some(dir) = simple_dir {
            if !self.is_danger_ahead(dir) {
                return Some(dir);
            }
        }
        
        // Last resort: find any safe direction
        for direction in [Direction::Up, Direction::Down, Direction::Left, Direction::Right].iter() {
            if !self.is_danger_ahead(*direction) {
                // Check if it's not opposite to current direction
                match (&self.snake.direction, direction) {
                    (Direction::Up, Direction::Down) => continue,
                    (Direction::Down, Direction::Up) => continue,
                    (Direction::Left, Direction::Right) => continue,
                    (Direction::Right, Direction::Left) => continue,
                    _ => return Some(*direction),
                }
            }
        }
        
        None // No safe direction found
    }

    fn get_state(&self) -> Array1<f64> {
        let head = self.snake.get_head();
        let mut state = Array1::zeros(STATE_SIZE);

        // Snake head position (normalized)
        state[0] = head.x as f64 / GRID_WIDTH as f64;
        state[1] = head.y as f64 / GRID_HEIGHT as f64;

        // Food position (normalized)
        state[2] = self.food.position.x as f64 / GRID_WIDTH as f64;
        state[3] = self.food.position.y as f64 / GRID_HEIGHT as f64;

        // Direction to food (normalized)
        let dx = self.food.position.x as i32 - head.x as i32;
        let dy = self.food.position.y as i32 - head.y as i32;
        state[4] = dx as f64 / GRID_WIDTH as f64;
        state[5] = dy as f64 / GRID_HEIGHT as f64;

        // Distance to food (normalized)
        state[6] = self.distance_to_food() / (GRID_WIDTH.max(GRID_HEIGHT) as f64);

        // Danger detection in all directions (binary: 1 if danger, 0 if safe)
        state[7] = if self.is_danger_ahead(Direction::Up) { 1.0 } else { 0.0 };
        state[8] = if self.is_danger_ahead(Direction::Down) { 1.0 } else { 0.0 };
        state[9] = if self.is_danger_ahead(Direction::Left) { 1.0 } else { 0.0 };
        state[10] = if self.is_danger_ahead(Direction::Right) { 1.0 } else { 0.0 };

        // Current direction (one-hot encoded)
        state[11] = if matches!(self.snake.direction, Direction::Up) { 1.0 } else { 0.0 };
        state[12] = if matches!(self.snake.direction, Direction::Down) { 1.0 } else { 0.0 };
        state[13] = if matches!(self.snake.direction, Direction::Left) { 1.0 } else { 0.0 };
        state[14] = if matches!(self.snake.direction, Direction::Right) { 1.0 } else { 0.0 };

        // Snake length (normalized)
        state[15] = (self.snake.body.len() as f64 - 3.0) / 20.0; // Subtract initial length, normalize

        state
    }

    fn get_simple_state(&self) -> usize {
        let head = self.snake.get_head();
        let food = &self.food.position;
        
        // More detailed state representation with wall proximity
        let mut state = 0;
        
        // Food direction (relative to head) - 8 directions
        let dx = food.x as i32 - head.x as i32;
        let dy = food.y as i32 - head.y as i32;
        
        if dx > 0 && dy == 0 { state |= 1; }      // Right
        else if dx > 0 && dy > 0 { state |= 2; }  // Down-Right
        else if dx == 0 && dy > 0 { state |= 3; } // Down
        else if dx < 0 && dy > 0 { state |= 4; }  // Down-Left
        else if dx < 0 && dy == 0 { state |= 5; } // Left
        else if dx < 0 && dy < 0 { state |= 6; }  // Up-Left
        else if dx == 0 && dy < 0 { state |= 7; } // Up
        else if dx > 0 && dy < 0 { state |= 8; }  // Up-Right
        
        // Distance category (4 categories)
        let distance = self.distance_to_food();
        if distance <= 2.0 { state |= 16; }       // Very close
        else if distance <= 5.0 { state |= 32; }  // Close
        else if distance <= 10.0 { state |= 48; } // Medium
        else { state |= 64; }                     // Far
        
        // Wall proximity (new!) - 4 bits for how close to each wall
        if head.x <= 2 { state |= 128; }          // Close to left wall
        if head.x >= GRID_WIDTH - 3 { state |= 256; }  // Close to right wall
        if head.y <= 2 { state |= 512; }         // Close to top wall
        if head.y >= GRID_HEIGHT - 3 { state |= 1024; } // Close to bottom wall
        
        // Danger detection in all directions (4 bits)
        if self.is_danger_ahead(Direction::Up) { state |= 2048; }
        if self.is_danger_ahead(Direction::Down) { state |= 4096; }
        if self.is_danger_ahead(Direction::Left) { state |= 8192; }
        if self.is_danger_ahead(Direction::Right) { state |= 16384; }
        
        // Current direction (4 possibilities)
        match self.snake.direction {
            Direction::Up => state |= 32768,
            Direction::Down => state |= 65536,
            Direction::Left => state |= 131072,
            Direction::Right => state |= 262144,
        }
        
        // Snake length category (3 bits for 8 categories)
        let length = self.snake.body.len();
        if length <= 5 { state |= 524288; }
        else if length <= 10 { state |= 1048576; }
        else if length <= 20 { state |= 1572864; }
        else { state |= 2097152; }
        
        state
    }

    fn is_danger_ahead(&self, direction: Direction) -> bool {
        let head = self.snake.get_head();
        
        // Calculate new position using signed integers to avoid overflow
        let (new_x, new_y) = match direction {
            Direction::Up => (head.x as i32, head.y as i32 - 1),
            Direction::Down => (head.x as i32, head.y as i32 + 1),
            Direction::Left => (head.x as i32 - 1, head.y as i32),
            Direction::Right => (head.x as i32 + 1, head.y as i32),
        };

        // Check boundary collision first
        if new_x < 0 || new_x >= GRID_WIDTH as i32 || new_y < 0 || new_y >= GRID_HEIGHT as i32 {
            return true;
        }

        // Convert to unsigned for body collision check
        let next_pos = Position {
            x: new_x as u32,
            y: new_y as u32,
        };

        // Check collision with body
        for segment in self.snake.body.iter().skip(1) {
            if next_pos.x == segment.x && next_pos.y == segment.y {
                return true;
            }
        }

        false
    }

    fn handle_input(&mut self, key: Key) {
        if self.game_over {
            if key == Key::Space {
                self.reset();
            }
            return;
        }

        match key {
            Key::Up => self.snake.update_direction(Direction::Up),
            Key::Down => self.snake.update_direction(Direction::Down),
            Key::Left => self.snake.update_direction(Direction::Left),
            Key::Right => self.snake.update_direction(Direction::Right),
            _ => {}
        }
    }

    fn make_action(&mut self, action: usize) -> f64 {
        let direction = Direction::from_action(action);
        self.snake.update_direction(direction);
        self.update()
    }

    fn make_astar_action(&mut self) -> f64 {
        if let Some(direction) = self.get_intelligent_direction() {
            self.snake.update_direction(direction);
        }
        self.update()
    }

    fn render(&self, context: Context, graphics: &mut G2d) {
        // Clear the screen
        clear([0.0, 0.0, 0.0, 1.0], graphics);

        // Draw boundary walls
        let wall_color = [0.5, 0.5, 0.5, 1.0]; // Gray walls
        let wall_thickness = 2.0;
        
        // Top wall
        rectangle(wall_color, [0.0, 0.0, 600.0, wall_thickness], context.transform, graphics);
        // Bottom wall  
        rectangle(wall_color, [0.0, 600.0 - wall_thickness, 600.0, wall_thickness], context.transform, graphics);
        // Left wall
        rectangle(wall_color, [0.0, 0.0, wall_thickness, 600.0], context.transform, graphics);
        // Right wall
        rectangle(wall_color, [600.0 - wall_thickness, 0.0, wall_thickness, 600.0], context.transform, graphics);

        // Draw stats panel background
        rectangle(
            [0.15, 0.15, 0.15, 1.0], // Dark gray background
            [600.0, 0.0, STATS_PANEL_WIDTH, WINDOW_HEIGHT],
            context.transform,
            graphics,
        );

        // Draw game area border
        rectangle(
            [0.3, 0.3, 0.3, 1.0], // Gray border
            [598.0, 0.0, 4.0, WINDOW_HEIGHT], // Vertical border
            context.transform,
            graphics,
        );

        // Draw snake
        for (i, segment) in self.snake.body.iter().enumerate() {
            let color = if i == 0 {
                [0.0, 0.8, 0.0, 1.0] // Darker green for head
            } else {
                [0.0, 1.0, 0.0, 1.0] // Bright green for body
            };
            
            rectangle(
                color,
                [
                    segment.x as f64 * GRID_SIZE,
                    segment.y as f64 * GRID_SIZE,
                    GRID_SIZE,
                    GRID_SIZE,
                ],
                context.transform,
                graphics,
            );
        }

        // Draw food
        rectangle(
            [1.0, 0.0, 0.0, 1.0], // Red color
            [
                self.food.position.x as f64 * GRID_SIZE,
                self.food.position.y as f64 * GRID_SIZE,
                GRID_SIZE,
                GRID_SIZE,
            ],
            context.transform,
            graphics,
        );

        // Draw stats text (simplified - in a real implementation you'd use a font library)
        // For now, we'll draw colored rectangles to represent stats
        self.draw_stats_panel(context, graphics);

        // Draw game over overlay
        if self.game_over {
            rectangle(
                [1.0, 1.0, 1.0, 0.8], // Semi-transparent white
                [300.0 - 100.0, WINDOW_HEIGHT / 2.0 - 50.0, 200.0, 100.0],
                context.transform,
                graphics,
            );
        }
    }

    fn draw_stats_panel(&self, context: Context, graphics: &mut G2d) {
        let x_offset = 620.0;
        let mut y_offset = 30.0;
        let line_height = 60.0;
        let label_height = 20.0;
        let number_height = 35.0;
        let panel_width = 160.0;

        // Score Section
        // Draw "SCORE" title
        self.draw_title("SCORE", x_offset, y_offset, context, graphics);
        y_offset += label_height + 5.0;

        // Score value background
        rectangle(
            [0.0, 0.6, 0.0, 1.0], // Dark green
            [x_offset, y_offset, panel_width, number_height],
            context.transform,
            graphics,
        );

        // Score number as visual bars
        let score_str = format!("{}", self.score);
        let digit_width = panel_width / score_str.len() as f64;
        for (i, digit_char) in score_str.chars().enumerate() {
            let digit = digit_char.to_digit(10).unwrap_or(0) as f64;
            let bar_height = (digit / 9.0) * (number_height - 4.0);
            rectangle(
                [0.0, 1.0, 0.0, 1.0], // Bright green
                [x_offset + 2.0 + i as f64 * digit_width, 
                 y_offset + number_height - bar_height - 2.0, 
                 digit_width - 4.0, 
                 bar_height],
                context.transform,
                graphics,
            );
        }
        y_offset += number_height + line_height;

        // Hits Section
        let hits = self.points_table.current_session.hits;
        
        // Draw "HITS" title
        self.draw_title("HITS", x_offset, y_offset, context, graphics);
        y_offset += label_height + 5.0;

        // Hits value background
        rectangle(
            [0.0, 0.4, 0.6, 1.0], // Dark cyan
            [x_offset, y_offset, panel_width, number_height],
            context.transform,
            graphics,
        );

        // Hits number as visual bars
        let hits_str = format!("{}", hits);
        let digit_width = panel_width / hits_str.len() as f64;
        for (i, digit_char) in hits_str.chars().enumerate() {
            let digit = digit_char.to_digit(10).unwrap_or(0) as f64;
            let bar_height = (digit / 9.0) * (number_height - 4.0);
            rectangle(
                [0.0, 0.8, 1.0, 1.0], // Bright cyan
                [x_offset + 2.0 + i as f64 * digit_width, 
                 y_offset + number_height - bar_height - 2.0, 
                 digit_width - 4.0, 
                 bar_height],
                context.transform,
                graphics,
            );
        }
        y_offset += number_height + line_height;

        // Misses Section
        let misses = self.points_table.current_session.misses;
        
        // Draw "MISSES" title
        self.draw_title("MISSES", x_offset, y_offset, context, graphics);
        y_offset += label_height + 5.0;

        // Misses value background
        rectangle(
            [0.6, 0.2, 0.0, 1.0], // Dark orange-red
            [x_offset, y_offset, panel_width, number_height],
            context.transform,
            graphics,
        );

        // Misses number as visual bars (show only last 6 digits for readability)
        let misses_display = if misses > 999999 { misses / 1000 } else { misses };
        let misses_str = format!("{}", misses_display);
        let digit_width = panel_width / misses_str.len() as f64;
        for (i, digit_char) in misses_str.chars().enumerate() {
            let digit = digit_char.to_digit(10).unwrap_or(0) as f64;
            let bar_height = (digit / 9.0) * (number_height - 4.0);
            rectangle(
                [1.0, 0.4, 0.0, 1.0], // Orange-red
                [x_offset + 2.0 + i as f64 * digit_width, 
                 y_offset + number_height - bar_height - 2.0, 
                 digit_width - 4.0, 
                 bar_height],
                context.transform,
                graphics,
            );
        }

        // Simple legend at the bottom
        let legend_y = WINDOW_HEIGHT - 140.0;
        let legend_items = [
            ([0.0, 1.0, 0.0, 1.0], "SCORE"),
            ([0.0, 0.8, 1.0, 1.0], "HITS"),
            ([1.0, 0.4, 0.0, 1.0], "MISSES"),
        ];

        for (i, (color, label)) in legend_items.iter().enumerate() {
            let legend_y_pos = legend_y + i as f64 * 30.0;
            
            // Color indicator (larger)
            rectangle(
                *color,
                [x_offset, legend_y_pos, 25.0, 18.0],
                context.transform,
                graphics,
            );
            
            // Draw text label (larger and more spaced)
            self.draw_legend_text(label, x_offset + 30.0, legend_y_pos + 2.0, context, graphics);
        }
    }

    fn draw_title(&self, text: &str, x: f64, y: f64, context: Context, graphics: &mut G2d) {
        let char_width = 12.0;
        let char_height = 16.0;
        let spacing = 2.0;
        
        for (i, ch) in text.chars().enumerate() {
            let char_x = x + i as f64 * (char_width + spacing);
            self.draw_char(ch, char_x, y, char_width, char_height, [1.0, 1.0, 1.0, 1.0], context, graphics);
        }
    }

    fn draw_small_text(&self, text: &str, x: f64, y: f64, context: Context, graphics: &mut G2d) {
        let char_width = 8.0;
        let char_height = 11.0;
        let spacing = 1.0;
        
        for (i, ch) in text.chars().enumerate() {
            let char_x = x + i as f64 * (char_width + spacing);
            self.draw_char(ch, char_x, y, char_width, char_height, [0.8, 0.8, 0.8, 1.0], context, graphics);
        }
    }

    fn draw_legend_text(&self, text: &str, x: f64, y: f64, context: Context, graphics: &mut G2d) {
        let char_width = 10.0;
        let char_height = 14.0;
        let spacing = 1.5;
        
        for (i, ch) in text.chars().enumerate() {
            let char_x = x + i as f64 * (char_width + spacing);
            self.draw_char(ch, char_x, y, char_width, char_height, [0.9, 0.9, 0.9, 1.0], context, graphics);
        }
    }

    fn draw_char(&self, ch: char, x: f64, y: f64, width: f64, height: f64, color: [f32; 4], context: Context, graphics: &mut G2d) {
        let pixel_size = 1.5;
        
        // Improved bitmap patterns for better readability
        let pattern = match ch {
            'S' => vec![
                " ████ ",
                "█     ",
                "█     ",
                " ████ ",
                "     █",
                "     █",
                " ████ ",
            ],
            'C' => vec![
                " ████ ",
                "█     ",
                "█     ",
                "█     ",
                "█     ",
                "█     ",
                " ████ ",
            ],
            'O' => vec![
                " ████ ",
                "█    █",
                "█    █",
                "█    █",
                "█    █",
                "█    █",
                " ████ ",
            ],
            'R' => vec![
                "█████ ",
                "█    █",
                "█    █",
                "█████ ",
                "██    ",
                "█ █   ",
                "█  █  ",
            ],
            'E' => vec![
                "██████",
                "█     ",
                "█     ",
                "█████ ",
                "█     ",
                "█     ",
                "██████",
            ],
            'H' => vec![
                "█    █",
                "█    █",
                "█    █",
                "██████",
                "█    █",
                "█    █",
                "█    █",
            ],
            'I' => vec![
                "██████",
                "  ██  ",
                "  ██  ",
                "  ██  ",
                "  ██  ",
                "  ██  ",
                "██████",
            ],
            'T' => vec![
                "██████",
                "  ██  ",
                "  ██  ",
                "  ██  ",
                "  ██  ",
                "  ██  ",
                "  ██  ",
            ],
            'M' => vec![
                "█    █",
                "██  ██",
                "█ ██ █",
                "█    █",
                "█    █",
                "█    █",
                "█    █",
            ],
            _ => vec![
                "██████",
                "█    █",
                "█    █",
                "█    █",
                "█    █",
                "█    █",
                "██████",
            ],
        };

        for (row, line) in pattern.iter().enumerate() {
            for (col, pixel) in line.chars().enumerate() {
                if pixel == '█' {
                    rectangle(
                        color,
                        [x + col as f64 * pixel_size, y + row as f64 * pixel_size, pixel_size, pixel_size],
                        context.transform,
                        graphics,
                    );
                }
            }
        }
    }
}

// Serializable data structure for neural network
#[derive(Serialize, Deserialize)]
struct NetworkData {
    weights1_shape: (usize, usize),
    weights1_data: Vec<f64>,
    bias1_data: Vec<f64>,
    weights2_shape: (usize, usize),
    weights2_data: Vec<f64>,
    bias2_data: Vec<f64>,
    weights3_shape: (usize, usize),
    weights3_data: Vec<f64>,
    bias3_data: Vec<f64>,
}

// Improved Neural Network Implementation with proper backpropagation
#[derive(Clone)]
struct NeuralNetwork {
    weights1: Array2<f64>,
    bias1: Array1<f64>,
    weights2: Array2<f64>,
    bias2: Array1<f64>,
    weights3: Array2<f64>,
    bias3: Array1<f64>,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = thread_rng();
        // Xavier initialization for better learning
        let w1_scale = (2.0 / input_size as f64).sqrt();
        let w2_scale = (2.0 / hidden_size as f64).sqrt();
        let w3_scale = (2.0 / hidden_size as f64).sqrt();

        NeuralNetwork {
            weights1: Array2::from_shape_fn((input_size, hidden_size), |_| {
                Normal::new(0.0, w1_scale).unwrap().sample(&mut rng)
            }),
            bias1: Array1::zeros(hidden_size),
            weights2: Array2::from_shape_fn((hidden_size, hidden_size), |_| {
                Normal::new(0.0, w2_scale).unwrap().sample(&mut rng)
            }),
            bias2: Array1::zeros(hidden_size),
            weights3: Array2::from_shape_fn((hidden_size, output_size), |_| {
                Normal::new(0.0, w3_scale).unwrap().sample(&mut rng)
            }),
            bias3: Array1::zeros(output_size),
        }
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        // First layer
        let z1 = input.dot(&self.weights1) + &self.bias1;
        let a1 = z1.map(|x| x.max(0.0)); // ReLU activation

        // Second layer
        let z2 = a1.dot(&self.weights2) + &self.bias2;
        let a2 = z2.map(|x| x.max(0.0)); // ReLU activation

        // Output layer
        a2.dot(&self.weights3) + &self.bias3
    }

    fn train(&mut self, states: &[Array1<f64>], targets: &[Array1<f64>]) {
        let lr = LEARNING_RATE;
        let batch_size = states.len() as f64;

        // Initialize gradients
        let mut grad_w1: Array2<f64> = Array2::zeros(self.weights1.dim());
        let mut grad_b1: Array1<f64> = Array1::zeros(self.bias1.len());
        let mut grad_w2: Array2<f64> = Array2::zeros(self.weights2.dim());
        let mut grad_b2: Array1<f64> = Array1::zeros(self.bias2.len());
        let mut grad_w3: Array2<f64> = Array2::zeros(self.weights3.dim());
        let mut grad_b3: Array1<f64> = Array1::zeros(self.bias3.len());

        for (state, target) in states.iter().zip(targets.iter()) {
            // Forward pass
            let z1 = state.dot(&self.weights1) + &self.bias1;
            let a1 = z1.map(|x| x.max(0.0));
            let z2 = a1.dot(&self.weights2) + &self.bias2;
            let a2 = z2.map(|x| x.max(0.0));
            let output = a2.dot(&self.weights3) + &self.bias3;

            // Backward pass
            let delta3 = &output - target;

            // Output layer gradients
            for i in 0..self.weights3.nrows() {
                for j in 0..self.weights3.ncols() {
                    grad_w3[[i, j]] += delta3[j] * a2[i];
                }
            }
            grad_b3 = &grad_b3 + &delta3;

            // Second hidden layer gradients
            let delta2 = self.weights3.dot(&delta3);
            let delta2_relu = Array1::from_iter(
                delta2.iter().zip(z2.iter()).map(|(&d, &z)| if z > 0.0 { d } else { 0.0 })
            );

            for i in 0..self.weights2.nrows() {
                for j in 0..self.weights2.ncols() {
                    grad_w2[[i, j]] += delta2_relu[j] * a1[i];
                }
            }
            grad_b2 = &grad_b2 + &delta2_relu;

            // First hidden layer gradients
            let delta1 = self.weights2.dot(&delta2_relu);
            let delta1_relu = Array1::from_iter(
                delta1.iter().zip(z1.iter()).map(|(&d, &z)| if z > 0.0 { d } else { 0.0 })
            );

            for i in 0..self.weights1.nrows() {
                for j in 0..self.weights1.ncols() {
                    grad_w1[[i, j]] += delta1_relu[j] * state[i];
                }
            }
            grad_b1 = &grad_b1 + &delta1_relu;
        }

        // Update weights with averaged gradients
        self.weights1 = &self.weights1 - &(grad_w1 * (lr / batch_size));
        self.bias1 = &self.bias1 - &(grad_b1 * (lr / batch_size));
        self.weights2 = &self.weights2 - &(grad_w2 * (lr / batch_size));
        self.bias2 = &self.bias2 - &(grad_b2 * (lr / batch_size));
        self.weights3 = &self.weights3 - &(grad_w3 * (lr / batch_size));
        self.bias3 = &self.bias3 - &(grad_b3 * (lr / batch_size));
    }

    fn copy_from(&mut self, other: &NeuralNetwork) {
        self.weights1 = other.weights1.clone();
        self.bias1 = other.bias1.clone();
        self.weights2 = other.weights2.clone();
        self.bias2 = other.bias2.clone();
        self.weights3 = other.weights3.clone();
        self.bias3 = other.bias3.clone();
    }

    fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Convert arrays to vectors for serialization
        let data = NetworkData {
            weights1_shape: self.weights1.dim(),
            weights1_data: self.weights1.iter().cloned().collect(),
            bias1_data: self.bias1.iter().cloned().collect(),
            weights2_shape: self.weights2.dim(),
            weights2_data: self.weights2.iter().cloned().collect(),
            bias2_data: self.bias2.iter().cloned().collect(),
            weights3_shape: self.weights3.dim(),
            weights3_data: self.weights3.iter().cloned().collect(),
            bias3_data: self.bias3.iter().cloned().collect(),
        };
        let json = serde_json::to_string_pretty(&data)?;
        fs::write(filename, json)?;
        Ok(())
    }

    fn load(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(filename)?;
        let data: NetworkData = serde_json::from_str(&json)?;
        
        let weights1 = Array2::from_shape_vec(data.weights1_shape, data.weights1_data)?;
        let bias1 = Array1::from_vec(data.bias1_data);
        let weights2 = Array2::from_shape_vec(data.weights2_shape, data.weights2_data)?;
        let bias2 = Array1::from_vec(data.bias2_data);
        let weights3 = Array2::from_shape_vec(data.weights3_shape, data.weights3_data)?;
        let bias3 = Array1::from_vec(data.bias3_data);
        
        Ok(NeuralNetwork {
            weights1,
            bias1,
            weights2,
            bias2,
            weights3,
            bias3,
        })
    }
}

// Simple Q-Learning Agent (much faster learning)
struct SimpleQAgent {
    q_table: std::collections::HashMap<usize, [f64; ACTION_SIZE]>,
    learning_rate: f64,
    discount_factor: f64,
    epsilon: f64,
    epsilon_decay: f64,
    epsilon_min: f64,
}

impl SimpleQAgent {
    fn new() -> Self {
        SimpleQAgent {
            q_table: std::collections::HashMap::new(),
            learning_rate: 0.1,
            discount_factor: 0.95,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
        }
    }

    fn get_q_values(&mut self, state: usize) -> &mut [f64; ACTION_SIZE] {
        self.q_table.entry(state).or_insert([0.0; ACTION_SIZE])
    }

    fn act(&mut self, state: usize) -> usize {
        if thread_rng().gen::<f64>() < self.epsilon {
            thread_rng().gen_range(0..ACTION_SIZE)
        } else {
            let q_values = self.get_q_values(state);
            q_values.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        }
    }

    fn act_with_astar_guidance(&mut self, state: usize, game: &mut Game, guidance_weight: f64) -> usize {
        // Get A* suggested action
        let astar_action = if let Some(direction) = game.get_astar_direction() {
            direction as usize
        } else {
            // Fallback to random if A* fails
            thread_rng().gen_range(0..ACTION_SIZE)
        };

        if thread_rng().gen::<f64>() < self.epsilon {
            // During exploration, bias towards A* suggestion
            if thread_rng().gen::<f64>() < guidance_weight {
                astar_action
            } else {
                thread_rng().gen_range(0..ACTION_SIZE)
            }
        } else {
            // During exploitation, combine Q-values with A* guidance
            let q_values = self.get_q_values(state);
            let mut guided_q_values = *q_values;
            
            // Boost the A* suggested action
            guided_q_values[astar_action] += guidance_weight;
            
            guided_q_values.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        }
    }

    fn learn(&mut self, state: usize, action: usize, reward: f64, next_state: usize, done: bool) {
        let current_q = self.get_q_values(state)[action];
        
        let next_max_q = if done {
            0.0
        } else {
            *self.get_q_values(next_state).iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        };

        let target = reward + self.discount_factor * next_max_q;
        let new_q = current_q + self.learning_rate * (target - current_q);
        
        self.get_q_values(state)[action] = new_q;
    }

    fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.q_table)?;
        fs::write(filename, json)?;
        Ok(())
    }

    fn load(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = fs::read_to_string(filename)?;
        self.q_table = serde_json::from_str(&json)?;
        self.epsilon = 0.0; // No exploration when loading trained model
        Ok(())
    }
}

#[derive(Clone)]
struct Experience {
    state: Array1<f64>,
    action: usize,
    reward: f64,
    next_state: Array1<f64>,
    done: bool,
}

struct DQNAgent {
    q_network: NeuralNetwork,
    target_network: NeuralNetwork,
    memory: VecDeque<Experience>,
    epsilon: f64,
    episode: usize,
}

impl DQNAgent {
    fn new() -> Self {
        let q_network = NeuralNetwork::new(STATE_SIZE, 128, ACTION_SIZE); // Larger network
        let mut target_network = NeuralNetwork::new(STATE_SIZE, 128, ACTION_SIZE);
        target_network.copy_from(&q_network); // Initialize with same weights
        
        DQNAgent {
            q_network,
            target_network,
            memory: VecDeque::with_capacity(MEMORY_SIZE),
            epsilon: EPSILON_START,
            episode: 0,
        }
    }

    fn act(&self, state: &Array1<f64>) -> usize {
        if thread_rng().gen::<f64>() < self.epsilon {
            // Random action (exploration)
            thread_rng().gen_range(0..ACTION_SIZE)
        } else {
            // Greedy action (exploitation)
            let q_values = self.q_network.forward(state);
            q_values.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        }
    }

    fn remember(&mut self, experience: Experience) {
        if self.memory.len() >= MEMORY_SIZE {
            self.memory.pop_front();
        }
        self.memory.push_back(experience);
    }

    fn replay(&mut self) {
        if self.memory.len() < BATCH_SIZE {
            return;
        }

        // Sample random batch
        let mut batch = Vec::new();
        for _ in 0..BATCH_SIZE {
            let idx = thread_rng().gen_range(0..self.memory.len());
            batch.push(self.memory[idx].clone());
        }

        let mut states = Vec::new();
        let mut targets = Vec::new();

        for experience in batch {
            let mut target = self.q_network.forward(&experience.state);
            
            if experience.done {
                target[experience.action] = experience.reward;
            } else {
                let next_q_values = self.target_network.forward(&experience.next_state);
                let max_next_q = next_q_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                target[experience.action] = experience.reward + GAMMA * max_next_q;
            }

            states.push(experience.state);
            targets.push(target);
        }

        self.q_network.train(&states, &targets);
    }

    fn update_target_network(&mut self) {
        // Properly copy weights from main network to target network
        self.target_network.copy_from(&self.q_network);
    }

    fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * EPSILON_DECAY).max(EPSILON_END);
        self.episode += 1;
    }

    fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.q_network.save(filename)
    }

    fn load(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.q_network = NeuralNetwork::load(filename)?;
        self.target_network = NeuralNetwork::load(filename)?;
        Ok(())
    }
}

fn train_simple_agent(episodes: usize) {
    let mut agent = SimpleQAgent::new();
    let mut scores = Vec::new();
    
    println!("Starting simple Q-learning training for {} episodes...", episodes);
    println!("Episode | Score | Hits | Misses | Hit Rate | Avg Score | Epsilon | Q-States");
    println!("--------|-------|------|--------|----------|----------|---------|----------");
    
    for episode in 0..episodes {
        let mut game = Game::new();
        let mut state = game.get_simple_state();
        let mut steps = 0;

        while !game.game_over && steps < 500 {
            let action = agent.act(state);
            let reward = game.make_action(action);
            let next_state = game.get_simple_state();
            
            agent.learn(state, action, reward, next_state, game.game_over);
            
            state = next_state;
            steps += 1;
        }

        agent.decay_epsilon();
        scores.push(game.score);

        if episode % 50 == 0 {
            let avg_score = if episode == 0 { 
                game.score as f64 
            } else {
                scores.iter().rev().take(50.min(episode + 1)).sum::<u32>() as f64 / 50.0_f64.min(episode as f64 + 1.0)
            };
            let session = &game.points_table.current_session;
            println!("{:7} | {:5} | {:4} | {:6} | {:7.1}% | {:8.2} | {:7.3} | {:11}", 
                     episode, game.score, session.hits, session.misses, 
                     session.hit_rate(), avg_score, agent.epsilon, agent.q_table.len());
        }

        // Save model periodically
        if episode % 200 == 0 && episode > 0 {
            if let Err(e) = agent.save("simple_q_model.json") {
                println!("Failed to save Q-table: {}", e);
            }
        }
    }
    
    // Save final model
    if let Err(e) = agent.save("simple_q_model_final.json") {
        println!("Failed to save final Q-table: {}", e);
    } else {
        println!("Final Q-table saved!");
    }
    
    // Print final statistics
    println!("\n=== Simple Q-Learning Training Complete ===");
    if let Some(last_score) = scores.last() {
        println!("Final Score: {}", last_score);
    }
    let avg_score = scores.iter().sum::<u32>() as f64 / scores.len() as f64;
    println!("Average Score: {:.2}", avg_score);
    println!("Best Score: {}", scores.iter().max().unwrap_or(&0));
    println!("Q-Table Size: {}", agent.q_table.len());
}

fn train_astar_guided_agent(episodes: usize) {
    let mut agent = SimpleQAgent::new();
    let mut scores = Vec::new();
    
    println!("Starting A*-guided Q-learning training for {} episodes...", episodes);
    println!("This combines A* pathfinding expertise with Q-learning exploration!");
    println!("Episode | Score | Hits | Misses | Hit Rate | Avg Score | Epsilon | Guidance | Q-States");
    println!("--------|-------|------|--------|----------|----------|---------|----------|----------");
    
    for episode in 0..episodes {
        let mut game = Game::new();
        let mut state = game.get_simple_state();
        let mut steps = 0;
        
        // Decrease guidance weight over time (start high, end low)
        // This allows A* to guide early learning, then lets Q-learning take over
        let guidance_weight = (1.0 - (episode as f64 / episodes as f64)) * 2.0;
        let guidance_weight = guidance_weight.max(0.1); // Minimum guidance

        while !game.game_over && steps < 500 {
            let action = agent.act_with_astar_guidance(state, &mut game, guidance_weight);
            let reward = game.make_action(action);
            let next_state = game.get_simple_state();
            
            agent.learn(state, action, reward, next_state, game.game_over);
            
            state = next_state;
            steps += 1;
        }

        agent.decay_epsilon();
        scores.push(game.score);

        if episode % 50 == 0 {
            let avg_score = if episode == 0 { 
                game.score as f64 
            } else {
                scores.iter().rev().take(50.min(episode + 1)).sum::<u32>() as f64 / 50.0_f64.min(episode as f64 + 1.0)
            };
            let session = &game.points_table.current_session;
            println!("{:7} | {:5} | {:4} | {:6} | {:7.1}% | {:8.2} | {:7.3} | {:8.2} | {:11}", 
                     episode, game.score, session.hits, session.misses, 
                     session.hit_rate(), avg_score, agent.epsilon, guidance_weight, agent.q_table.len());
        }

        // Save model periodically
        if episode % 200 == 0 && episode > 0 {
            if let Err(e) = agent.save("astar_guided_q_model.json") {
                println!("Failed to save Q-table: {}", e);
            }
        }
    }
    
    // Save final model
    if let Err(e) = agent.save("astar_guided_q_model_final.json") {
        println!("Failed to save final Q-table: {}", e);
    } else {
        println!("Final A*-guided Q-table saved!");
    }
    
    // Print final statistics
    println!("\n=== A*-Guided Training Complete ===");
    if let Some(last_score) = scores.last() {
        println!("Final Score: {}", last_score);
    }
    let avg_score = scores.iter().sum::<u32>() as f64 / scores.len() as f64;
    println!("Average Score: {:.2}", avg_score);
    println!("Best Score: {}", scores.iter().max().unwrap_or(&0));
    println!("Q-Table Size: {}", agent.q_table.len());
}

fn train_agent(episodes: usize) {
    let mut agent = DQNAgent::new();
    let mut scores = Vec::new();
    
    println!("Starting training for {} episodes...", episodes);
    println!("Episode | Score | Hits | Misses | Hit Rate | Avg Score | Epsilon");
    println!("--------|-------|------|--------|----------|----------|--------");
    
    for episode in 0..episodes {
        let mut game = Game::new();
        let mut state = game.get_state();
        let mut _total_reward = 0.0;
        let mut steps = 0;

        while !game.game_over && steps < 1000 {
            let action = agent.act(&state);
            let reward = game.make_action(action);
            let next_state = game.get_state();
            
            agent.remember(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done: game.game_over,
            });

            state = next_state;
            _total_reward += reward;
            steps += 1;

            if steps % 2 == 0 && agent.memory.len() >= BATCH_SIZE {
                agent.replay();
            }
        }

        agent.decay_epsilon();
        scores.push(game.score);

        if episode % 100 == 0 {
            let avg_score = scores.iter().rev().take(100).sum::<u32>() as f64 / 100.0;
            let session = &game.points_table.current_session;
            println!("{:7} | {:5} | {:4} | {:6} | {:7.1}% | {:8.2} | {:6.3}", 
                     episode, game.score, session.hits, session.misses, 
                     session.hit_rate(), avg_score, agent.epsilon);
        }

        if episode % TARGET_UPDATE_FREQ == 0 && episode > 0 {
            agent.update_target_network();
        }

        if episode % 2000 == 0 && episode > 0 {
            if let Err(e) = agent.save("snake_model.json") {
                println!("Failed to save model: {}", e);
            } else {
                println!("Model saved at episode {}", episode);
            }
        }
    }

    if let Err(e) = agent.save("snake_model_final.json") {
        println!("Failed to save final model: {}", e);
    } else {
        println!("Final model saved!");
    }
    
    // Print final statistics
    println!("\n=== Training Complete ===");
    if let Some(last_score) = scores.last() {
        println!("Final Score: {}", last_score);
    }
    let avg_score = scores.iter().sum::<u32>() as f64 / scores.len() as f64;
    println!("Average Score: {:.2}", avg_score);
    println!("Best Score: {}", scores.iter().max().unwrap_or(&0));
}

fn run_astar_game() {
    let mut window: PistonWindow = WindowSettings::new("Snake Game - A* Pathfinding", [WINDOW_WIDTH, WINDOW_HEIGHT])
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut game = Game::new();
    let mut last_update = std::time::Instant::now();
    let update_interval = std::time::Duration::from_millis(200); // Slightly slower for better visualization
    let mut game_count = 0;

    println!("A* Pathfinding AI is playing! Press ESC to exit. Press SPACE to restart when game over.");
    println!("\nGame Statistics:");
    println!("Game | Score | Hits | Misses | Hit Rate");
    println!("-----|-------|------|--------|----------");

    while let Some(event) = window.next() {
        if let Some(Button::Keyboard(Key::Space)) = event.press_args() {
            if game.game_over {
                // Print game statistics before reset
                let session = &game.points_table.current_session;
                game_count += 1;
                println!("{:4} | {:5} | {:4} | {:6} | {:7.1}%", 
                         game_count, game.score, session.hits, session.misses, 
                         session.hit_rate());
                
                game.reset();
            }
        }

        if let Some(_) = event.update_args() {
            let now = std::time::Instant::now();
            if now.duration_since(last_update) >= update_interval {
                if !game.game_over {
                    game.make_astar_action(); // Use A* pathfinding
                } else {
                    // Print game statistics and auto-restart after a brief pause
                    let session = &game.points_table.current_session;
                    game_count += 1;
                    println!("{:4} | {:5} | {:4} | {:6} | {:7.1}%", 
                             game_count, game.score, session.hits, session.misses, 
                             session.hit_rate());
                    
                    std::thread::sleep(std::time::Duration::from_millis(2000));
                    game.reset();
                }
                last_update = now;
            }
        }

        if let Some(_args) = event.render_args() {
            window.draw_2d(&event, |context, graphics, _device| {
                game.render(context, graphics);
            });
        }
    }
}

fn main() {
    println!("Snake Game with RL Agent");
    println!("Commands:");
    println!("  cargo run -- train <episodes>     : Train the DQN agent");
    println!("  cargo run -- qtrain <episodes>    : Train the simple Q-learning agent (faster)");
    println!("  cargo run -- aguidedtrain <episodes> : Train Q-learning with A* guidance (best)");
    println!("  cargo run -- play                 : Play with trained agent");
    println!("  cargo run -- astar                : Play with A* pathfinding");
    println!("  cargo run -- human                : Play manually");
    println!("  cargo run                         : Default human play");

    let args: Vec<String> = std::env::args().collect();
    
    if args.len() > 1 {
        match args[1].as_str() {
            "train" => {
                let episodes = if args.len() > 2 {
                    args[2].parse().unwrap_or(5000)
                } else {
                    5000
                };
                train_agent(episodes);
                return;
            },
            "qtrain" => {
                let episodes = if args.len() > 2 {
                    args[2].parse().unwrap_or(1000)
                } else {
                    1000
                };
                train_simple_agent(episodes);
                return;
            },
            "aguidedtrain" => {
                let episodes = if args.len() > 2 {
                    args[2].parse().unwrap_or(1000)
                } else {
                    1000
                };
                train_astar_guided_agent(episodes);
                return;
            },
            "astar" => {
                println!("Running A* pathfinding AI...");
                run_astar_game();
                return;
            },
            "play" => {
                // Try to load A*-guided agent first, then SimpleQ, then DQN
                let mut simple_agent = SimpleQAgent::new();
                if simple_agent.load("astar_guided_q_model_final.json").is_ok() {
                    println!("Loaded A*-guided Q-Learning model!");
                    run_simple_ai_game(simple_agent);
                } else if simple_agent.load("astar_guided_q_model.json").is_ok() {
                    println!("Loaded A*-guided Q-Learning checkpoint model!");
                    run_simple_ai_game(simple_agent);
                } else if simple_agent.load("simple_q_model_final.json").is_ok() {
                    println!("Loaded Simple Q-Learning model!");
                    run_simple_ai_game(simple_agent);
                } else if simple_agent.load("simple_q_model.json").is_ok() {
                    println!("Loaded Simple Q-Learning checkpoint model!");
                    run_simple_ai_game(simple_agent);
                } else {
                    // Fallback to DQN
                    let mut agent = DQNAgent::new();
                    agent.epsilon = 0.0;
                    
                    if let Err(e) = agent.load("snake_model_final.json") {
                        println!("No trained models found: {}. Training an A*-guided Q-learning agent...", e);
                        train_astar_guided_agent(500);
                        let mut new_agent = SimpleQAgent::new();
                        let _ = new_agent.load("astar_guided_q_model_final.json");
                        run_simple_ai_game(new_agent);
                    } else {
                        println!("Loaded DQN model!");
                        run_ai_game(agent);
                    }
                }
                return;
            },
            "human" => {
                // Continue to human play mode
            },
            _ => {
                println!("Unknown command. Running in human mode.");
            }
        }
    }

    // Human play mode
    run_human_game();
}

fn run_simple_ai_game(mut agent: SimpleQAgent) {
    let mut window: PistonWindow = WindowSettings::new("Snake Game - Simple Q-Learning AI", [WINDOW_WIDTH, WINDOW_HEIGHT])
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut game = Game::new();
    let mut last_update = std::time::Instant::now();
    let update_interval = std::time::Duration::from_millis(150); // Slower for better visualization
    let mut game_count = 0;

    println!("Simple Q-Learning AI is playing! Press ESC to exit. Press SPACE to restart when game over.");
    println!("\nGame Statistics:");
    println!("Game | Score | Hits | Misses | Hit Rate");
    println!("-----|-------|------|--------|----------");

    while let Some(event) = window.next() {
        if let Some(Button::Keyboard(Key::Space)) = event.press_args() {
            if game.game_over {
                // Print game statistics before reset
                let session = &game.points_table.current_session;
                game_count += 1;
                println!("{:4} | {:5} | {:4} | {:6} | {:7.1}%", 
                         game_count, game.score, session.hits, session.misses, 
                         session.hit_rate());
                
                game.reset();
            }
        }

        if let Some(_) = event.update_args() {
            let now = std::time::Instant::now();
            if now.duration_since(last_update) >= update_interval {
                if !game.game_over {
                    let state = game.get_simple_state();
                    let action = agent.act(state);
                    game.make_action(action);
                } else {
                    // Print game statistics and auto-restart after a brief pause
                    let session = &game.points_table.current_session;
                    game_count += 1;
                    println!("{:4} | {:5} | {:4} | {:6} | {:7.1}%", 
                             game_count, game.score, session.hits, session.misses, 
                             session.hit_rate());
                    
                    std::thread::sleep(std::time::Duration::from_millis(2000));
                    game.reset();
                }
                last_update = now;
            }
        }

        if let Some(_args) = event.render_args() {
            window.draw_2d(&event, |context, graphics, _device| {
                game.render(context, graphics);
            });
        }
    }
}

fn run_ai_game(agent: DQNAgent) {
    let mut window: PistonWindow = WindowSettings::new("Snake Game - AI Playing", [WINDOW_WIDTH, WINDOW_HEIGHT])
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut game = Game::new();
    let mut last_update = std::time::Instant::now();
    let update_interval = std::time::Duration::from_millis(100); // Faster for AI
    let mut game_count = 0;

    println!("AI is playing! Press ESC to exit. Press SPACE to restart when game over.");
    println!("\nGame Statistics:");
    println!("Game | Score | Hits | Misses | Hit Rate | Total Moves");
    println!("-----|-------|------|--------|----------|------------");

    while let Some(event) = window.next() {
        if let Some(Button::Keyboard(Key::Space)) = event.press_args() {
            if game.game_over {
                // Print game statistics before reset
                let session = &game.points_table.current_session;
                game_count += 1;
                println!("{:4} | {:5} | {:4} | {:6} | {:7.1}%", 
                         game_count, game.score, session.hits, session.misses, 
                         session.hit_rate());
                
                game.reset();
            }
        }

        if let Some(_) = event.update_args() {
            let now = std::time::Instant::now();
            if now.duration_since(last_update) >= update_interval {
                if !game.game_over {
                    let state = game.get_state();
                    let action = agent.act(&state);
                    game.make_action(action);
                } else {
                    // Print game statistics and auto-restart after a brief pause
                    let session = &game.points_table.current_session;
                    game_count += 1;
                    println!("{:4} | {:5} | {:4} | {:6} | {:7.1}%", 
                             game_count, game.score, session.hits, session.misses, 
                             session.hit_rate());
                    
                    std::thread::sleep(std::time::Duration::from_millis(1000));
                    game.reset();
                }
                last_update = now;
            }
        }

        if let Some(_args) = event.render_args() {
            window.draw_2d(&event, |context, graphics, _device| {
                game.render(context, graphics);
            });
        }
    }
}

fn run_human_game() {
    let mut window: PistonWindow = WindowSettings::new("Snake Game - Human Player", [WINDOW_WIDTH, WINDOW_HEIGHT])
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut game = Game::new();
    let mut last_update = std::time::Instant::now();
    let update_interval = std::time::Duration::from_millis(150);

    while let Some(event) = window.next() {
        if let Some(Button::Keyboard(key)) = event.press_args() {
            game.handle_input(key);
        }

        if let Some(_) = event.update_args() {
            let now = std::time::Instant::now();
            if now.duration_since(last_update) >= update_interval {
                game.update();
                last_update = now;
            }
        }

        if let Some(_args) = event.render_args() {
            window.draw_2d(&event, |context, graphics, _device| {
                game.render(context, graphics);
            });
        }
    }
}
