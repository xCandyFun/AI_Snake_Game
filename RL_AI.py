import random
import numpy as np
from Snake_Game import SnakeGame, Direction, rotate_left, rotate_right

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

        point_l = game.move(rotate_left(game.direction)) # Left
        point_r = game.move(rotate_right(game.direction)) # Right
        point_s = game.move(game.direction) # Straight

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        food = game.food
        
        state = [
            # Danger straight
            game._is_collision(point_s),
            # Danger right
            game._is_collision(point_r),
            # Danger left
            game._is_collision(point_l),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            food.x < head.x, # food left
            food.x > head.x, # food right
            food.y < head.y, # food up
            food.y > head.y # food down
        ]

        return tuple(int(s) for s in state)
    
    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state][action]
    
    def update_q_value(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table.get(next_state, np.zeros(len(self.actions))))
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state][action] = new_q

    def choose_action(self, state):
        if random.uniform(0,1) < self.exploration_rate:
            return random.choice(range(len(self.actions))) # Random action (exploration)
        return np.argmax(self.q_table.get(state, np.zeros(len(self.actions)))) # Best action (exploitation)
    
    def decay_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay
