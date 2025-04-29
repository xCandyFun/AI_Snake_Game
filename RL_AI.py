import random
import numpy as np
from Snake_Game import SnakeGame

class QLearningAgent:

    def __init__(self, actions, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}

    def get_state(self, game):
        head = game.snake[0]
        food = game.food
        # Simplified state (x, y positions of the snake head and food)
        return (head.x, head.y, food.x, food.y)
    
    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state][action]
    
    def update_q_value(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table.get(next_state, np.zeros(len(self.actions))))
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state, action] = new_q

    def choose_action(self, state):
        if random.uniform(0,1) < self.exploration_rate:
            return random.choice(range(len(self.actions))) # Random action (exploration)
        return np.argmax(self.q_table.get(state, np.zeros(len(self.actions)))) # Best action (exploitation)
    
    def decay_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay
