from Snake_Game import SnakeGame
import random
import time

def random_action():
    return random.choice([[1, 0, 0], # move straight
                          [0, 1, 0], # tuen right
                          [0, 0, 1]]) # turn left

def main():
    game = SnakeGame()

    while True:
        action = random_action()
        reward, game_over, score = game.play_step(action)

        if game_over:
            print(f"Game Over! Your Score: {score}")
            time.sleep(1)
            game.reset() # Restart game after losing

if __name__ == "__main__":
    main()