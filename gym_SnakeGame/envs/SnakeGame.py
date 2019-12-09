import gym
import numpy as np
import warnings
import cv2

from gym import error, spaces, utils
from gym.utils import seeding

class SnakeGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=7):
        self.size = size
        self.state = np.zeros((size, size, 3)) # 0: item, 1: Snake Head, 2: Snake Body
        self.snake = list()
        self.snake.append((np.random.randint(0, self.size), np.random.randint(0, self.size)))
        self.item_pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        while self.item_pos == self.snake[0]:
            self.item_pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        self.length = 1
        self.need_new_tile = False
        self._update_tile()
        self.state_size = size * size * 3
        self.action_size = 4
        self.without_reward = 0

    def step(self, action):
        # 0: left, 1: up, 2: right, 3: down
        done = False
        reward = 0
        self.without_reward += 1

        if not self._valid_action(action):
            reward = 0
            done = True

        self.last_action = action

        if not done:
            x, y = self.snake[0]
            if action == 0:
                x -= 1
            elif action == 1:
                y -= 1
            elif action == 2:
                x += 1
            elif action == 3:
                y += 1
            else:
                warnings.warn(f"The action should be 0 or 1 or 2 or 3 but {action} was detected.")

            next_tile_type = np.argmax(self.state[x, y])

            if (self.state[x, y] == [0, 0, 0]).all():
                self.snake.insert(0, (x, y))
                del self.snake[-1]
            
            elif next_tile_type == 0:
                self.length += 1
                reward += 1
                self.without_reward = 0
                self.snake.insert(0, (x, y))
                self.need_new_tile = True

            elif next_tile_type == 1:
                reward = 0
                done = True
                warnings.warn(f"Next block must not be snake head. Something is wrong.")
            
            elif next_tile_type == 2:
                reward = 0
                done = True
            
            else:
                warnings.warn(f"The next tile type should be 0 or 1 or 2 or 3 but {next_tile_type} was detected.")

            if self.length == self.size * self.size - 1:
                done = True
                reward += 1

            else:
                self._update_tile()
        
        return self.state, reward, done, {}

    def reset(self):
        self.__init__(size=self.size)

        return self.state

    def render(self, mode='human', close=False):
        img_size = 600
        img_size = 600 - img_size % self.size
        tile_size = img_size // self.size
        padding_size = int(tile_size * 0.03)
        img = np.zeros((img_size, img_size, 3), np.uint8)

        cv2.rectangle(img, (0, 0), (img_size, img_size), (200, 200, 200), -1)

        for i in range(self.size):
            for j in range(self.size):
                cv2.rectangle(img, (i * tile_size + padding_size, j * tile_size + padding_size), ((i + 1) * tile_size - padding_size, (j + 1) * tile_size - padding_size), (10, 10, 10), -1)

        for i, snake_piece in enumerate(self.snake):
            x, y = snake_piece
            
            if i == 0:
                cv2.rectangle(img, (x * tile_size + padding_size, y * tile_size + padding_size), ((x + 1) * tile_size - padding_size, (y + 1) * tile_size - padding_size), (50, 150, 50), -1)
            else:
                cv2.rectangle(img, (x * tile_size + padding_size, y * tile_size + padding_size), ((x + 1) * tile_size - padding_size, (y + 1) * tile_size - padding_size), (100, 150, 100), -1)

        item_x, item_y = self.item_pos

        cv2.rectangle(img, (item_x * tile_size + padding_size, item_y * tile_size + padding_size), ((item_x + 1) * tile_size - padding_size, (item_y + 1) * tile_size - padding_size), (159, 150, 50), -1)

        return img

    def _update_tile(self):
        self.state[:, :, :] = 0

        for i, snake_piece in enumerate(self.snake):
            x, y = snake_piece
            if i == 0:
                self.state[x, y, 1] = 1
            else:
                self.state[x, y, 2] = 1

        if self.need_new_tile:
            self.need_new_tile = False
            while True:
                item_x, item_y = np.random.randint(0, self.size, 2)

                if (self.state[item_x, item_y] == [0, 0, 0]).all():
                    self.item_pos = (item_x, item_y)
                    break

        item_x, item_y = self.item_pos
        self.state[item_x, item_y, 0] = 1

    def _valid_action(self, action):
        x, y = self.snake[0]

        if action == 0:
            if x == 0:
                return False

        elif action == 1:
            if y == 0:
                return False

        elif action == 2:
            if x == (self.size - 1):
                return False
        
        elif action == 3:
            if y == (self.size - 1):
                return False
        else:
            warnings.warn(f"The action should be 0 or 1 or 2 or 3 but {action} was detected.")
            return False

        return True


if __name__ == "__main__":
    game = SnakeGameEnv()

    game.reset()
    for i in range(5):
        game.step(2)
        img = game.render()
        cv2.imshow("Snake Game", img)
        cv2.waitKey(500)
        game.step(3)
        img = game.render()
        cv2.imshow("Snake Game", img)
        cv2.waitKey(500)

    cv2.destroyAllWindows()