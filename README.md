# 🐍 Rust Snake Game with AI/ML Algorithms

A sophisticated Snake game implementation in Rust featuring multiple AI algorithms including **A* pathfinding**, **Deep Q-Networks (DQN)**, **Q-learning**, and a novel **A*-guided Q-learning** hybrid approach.

## 🎯 Usage Commands

### 🎮 Play Modes
```bash
cargo run                    # Human play (default)
cargo run -- human          # Human play (explicit)
cargo run -- play           # Play with trained AI agent
cargo run -- astar          # A* pathfinding AI (deterministic)
```

### 🧠 Training Modes
```bash
cargo run -- train 5000           # Train Deep Q-Network
cargo run -- qtrain 1000          # Train Simple Q-Learning
cargo run -- aguidedtrain 1000    # Train A*-Guided Q-Learning ⭐
```

## 🔬 Algorithm Details

### 1. A* Pathfinding
- Classic A* with Manhattan distance heuristic
- Uses BinaryHeap for priority queue
- Optimal pathfinding with obstacle avoidance

### 2. Deep Q-Network (DQN)
- 3-layer neural network (16→128→128→4)
- Experience replay with target networks
- ReLU activation, gradient descent optimization

### 3. Simple Q-Learning
- Tabular Q-learning with HashMap storage
- Fast convergence for discrete state space
- Classic Bellman equation updates

### 4. A*-Guided Q-Learning ⭐ Novel Innovation
- Hybrid approach combining A* optimality with Q-learning adaptability
- A* provides guidance that decreases over time
- Faster convergence than pure Q-learning
- Smooth transition from deterministic to learned behavior

## 🚀 Quick Start
```bash
# 1. Train an AI agent (5-10 minutes)
cargo run -- aguidedtrain 1000

# 2. Watch it play
cargo run -- play

# 3. Compare with A* pathfinding
cargo run -- astar
```

## 🔧 Technical Implementation
- **Pure Rust**: No external ML frameworks
- **Custom Neural Networks**: Hand-implemented backpropagation
- **2,125 lines of code**: Complete implementation from scratch
- **Real-time Stats**: Custom bitmap font rendering
