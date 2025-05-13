from Snake_Game import SnakeGame
from RL_AI import QLearningAgent
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pickle

def plot(scores):
        clear_output(wait=True)
        plt.figure(figsize=(10,5))
        plt.plot(scores, label='Score')
        plt.plot([sum(scores[max(0, i-50):i+1])/len(scores[max(0, i-50):i+1]) for i in range(len(scores))], label='Moving average (50)')
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Training Progress")
        plt.legend()
        plt.show()

def train():
    total_episodes = 500
    actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Straight, Right, Left
    agent = QLearningAgent(actions)
    scores = []
    

    for episode in range(total_episodes):
        game = SnakeGame()
        state = agent.get_state(game)
        total_reward = 0

        while True:
            action_idx = agent.choose_action(state)
            action = actions[action_idx]

            reward, game_over, score = game.play_step(action)
            next_state = agent.get_state(game)

            agent.update_q_value(state, action_idx, reward, next_state)
            state = next_state
            total_reward += reward
            
            if game_over:
                break

        agent.decay_exploration_rate()
        scores.append(score)
        print(f"Episode: {episode + 1}/{total_episodes}, Score: {score}, Epsilon: {agent.exploration_rate:.4f}")

    # Final plot after training
    plot(scores)

    
    #Save learned Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
    print("Q-table saved.")

if __name__ == "__main__":
    train()
        
        