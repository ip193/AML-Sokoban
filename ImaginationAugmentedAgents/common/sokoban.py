import gym_sokoban
from gym import spaces

from common.sokoban_env import SokobanEnv

gym_sokoban  # PyCharm hack


class Sokoban:
    def __init__(self, dim_room=(7, 7), max_steps=120, num_boxes=2, shape=(3, 7, 7), render_mode='tiny_rgb_array'):
        self.render_mode = render_mode

        self.env = SokobanEnv(dim_room=dim_room, max_steps=max_steps, num_boxes=num_boxes, render_mode=render_mode)
        self.done = False

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=shape)

    def step(self, action):
        observation, reward, self.done, info = self.env.step(action + 1)  # ignore NOP
        return observation.transpose(2, 0, 1), reward, self.done, info

    def reset(self):
        observation = self.env.reset()
        self.done = False
        return observation.transpose(2, 0, 1)
