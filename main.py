from Snake_Game import SnakeGame
from RL_AI import QLearningAgent
    

def main():

    actions = [[1, 0, 0], # move straight
               [0, 1, 0], # tuen right
               [0, 0, 1]] # turn left
    
    agent = QLearningAgent(actions)
    game = SnakeGame()

    episodes = 1000 # Number of training episodes

    for episode in range(episodes):
        game.reset()
        state = agent.get_state(game)
        total_reward = 0
        game_over = False

        while not game_over:
            action_idx = agent.choose_action(state)
            action = actions[action_idx]

            reward, game_over, score = game.play_step(action)
            total_reward += reward
            next_state = agent.get_state(game)

            agent.update_q_value(state, action_idx, reward, next_state)

            state = next_state

        agent.decay_exploration_rate()
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Exploration Rate {agent.exploration_rate}")

if __name__ == "__main__":
    main()