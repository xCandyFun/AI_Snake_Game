import pygame
import random
from enum import Enum
from collections import namedtuple

#Initialize pygame
pygame.init()

#Fonts
font = pygame.font.Font(None, 25)

# Direction Enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Size
BLOCK_SIZE = 20
SPEED = 10 # Frames per second

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Point
Point = namedtuple('Point', 'x, y')

class SnakeGame:

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head, 
                      Point(self.head.x- BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2* BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Move
        self._move(action) # Update the head
        self.snake.insert(0, self.head)

        # Check if game over
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # Return game info
        return reward, game_over, self.score
    
    def _is_collision(self, pt=None):
        
        if pt is None:
            pt = self.head
        
        # Hits boundary
        if pt.x > self.width - BLOCK_SIZE or pt.x < 0 or pt.y > self.height - BLOCK_SIZE or pt.y < 0:
            return True
        
        # Hits itself
        if pt in self.snake[1:]:
            return True
        
        return False  
    
    def move(self, direction):
        x = self.snake[0].x
        y = self.snake[0].y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        
        return Point(x,y)
         
    
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right turn, left turn]
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if action == [1, 0, 0]: # Straight
            new_dir = clockwise[idx]
        elif action == [0, 1, 0]: # Right turn
            new_dir = clockwise[(idx + 1) % 4]
        else: # Left turn
            new_dir = clockwise[(idx - 1) % 4]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

def rotate_left(direction):
    if direction == Direction.RIGHT:
        return Direction.UP
    if direction == Direction.UP:
        return Direction.LEFT
    if direction == Direction.LEFT:
        return Direction.DOWN
    if direction == Direction.DOWN:
        return Direction.RIGHT
    
def rotate_right(direction):
    if direction == Direction.RIGHT:
        return Direction.DOWN
    if direction == Direction.DOWN:
        return Direction.LEFT
    if direction == Direction.LEFT:
        return Direction.UP
    if direction == Direction.UP:
        return Direction.RIGHT
