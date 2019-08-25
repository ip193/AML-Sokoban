import gym_sokoban
from gym import spaces

from common.sokoban_env import SokobanEnv

gym_sokoban  # PyCharm hack


class Sokoban:
    def __init__(self):
        self.env = SokobanEnv(dim_room=(7, 7), max_steps=120, num_boxes=2)
        self.done = False

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(3, 7, 7))

    def step(self, action):
        observation, reward, self.done, info = self.env.step(action + 1, observation_mode='tiny_rgb_array')  # ignore NOP
        return observation.transpose(2, 0, 1), reward, self.done, info

    def reset(self):
        observation = self.env.reset(render_mode='tiny_rgb_array')
        self.done = False
        return observation.transpose(2, 0, 1)
