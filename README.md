# 🐍 Reinforcement Learning in the Snake Game

This project is an implementation of an AI agent that learns to play Snake using **Q-learning**, a classic reinforcement learning algorithm.

## 📁 Project Structure

- `Snake_Game.py` – Game logic and environment
- `RL_AI.py` – Q-learning agent
- `train.py` – Script for training the agent
- `play.py` – Script for testing the trained agent
- `q_table.pkl` – Saved Q-table after training
- `rapport_step2.pdf` – Report for Step 2 of the project

## 🧠 Model Overview

- **Algorithm:** Q-learning
- **State Representation:** 11 binary features representing the current game state (e.g., danger ahead/right/left, current direction, food location)
- **Actions:**  
  - `[1, 0, 0]` = Move forward  
  - `[0, 1, 0]` = Turn right  
  - `[0, 0, 1]` = Turn left
- **Reward System:**
  - `+10` for eating food  
  - `-10` for collisions (death)  
  - `0` for neutral moves

## 🚀 Getting Started

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python train.py
```
> This will train the model over 1000 episodes and save the Q-table as `q_table.pkl`.

### Play the Game with the Trained Agent
```bash
python play.py
```
> Loads the saved Q-table and lets the AI play autonomously without exploration.

## 📊 Evaluation

- Training results are visualized using a line chart showing the score per episode.
- The average score improves over time, indicating that the agent is learning effective strategies.

## 👤 Author

- ME
