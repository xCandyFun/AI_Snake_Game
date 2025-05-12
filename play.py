import pickle
from Snake_Game import SnakeGame
from RL_AI import QLearningAgent

def play():
    actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    agent = QLearningAgent(actions, exploration_rate=0.0)

    with open("q_table.pkl", "rb") as f:
        agent.q_table = pickle.load(f)

    game = SnakeGame()
    state = agent.get_state(game)

    while True:
        action_idx = agent.choose_action(state)
        action = actions[action_idx]
        reward, game_over, score = game.play_step(action)
        next_state = agent.get_state(game)
        state = next_state

        if game_over:
            print("Game Over! Final Score:", score)
            break

if __name__ == "__main__":
    play()