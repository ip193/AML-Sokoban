import gzip
import pickle
import time
import os
from random import randint

import numpy as np
from gym_sokoban.envs import SokobanEnv


def generate_env():
    return SokobanEnv(num_boxes=3, max_steps=200, reset=False)


def set_env_state(env, room_structures, states, idx):
    env.room_fixed = room_structures[idx]
    env.room_state = states[idx]
    env.room_state[env.room_state == 3] = 4
    # env.box_mapping = get_box_mapping(env.room_state)
    player_position = np.where(env.room_state == 5)
    env.player_position = np.asarray([player_position[0][0], player_position[1][0]])

    env.num_env_steps = 0
    env.reward_last = 0
    env.boxes_on_target = 0
    return env


def solve_game(env, actions, distances, idx, render_mode='human'):
    score = 0
    ACTION_LOOKUP = env.unwrapped.get_action_lookup()
    done = False

    for t in range(distances[idx]):
        env.render(mode=render_mode)

        action = actions[idx - t] + 1  # ignore 0 = no operation

        time.sleep(1)  # FIXME
        observation, reward, done, info = env.step(action)
        score += reward
        if render_mode == 'human':
            print(f'do {ACTION_LOOKUP[action]:10}, now {distances[idx - t] - 1:2} steps to go')
        if done:
            env.render(mode=render_mode)
            if render_mode == 'human':
                print('👌', "Episode finished after {} timesteps".format(t + 1))
            break
    if render_mode == 'human':
        print(score)
    env.close()

    return done


if __name__ == '__main__':

    jakob = True # FIXME

    if jakob:
        os.chdir("C:/Users/ASUS-N55S-Laptop/Desktop/AML Final Project/AML-Sokoban/data/")

    with gzip.open('../data/train/room_structures_train.pkl.gz', 'rb') as f:
        room_structures = pickle.load(f)

    with gzip.open('../data/train/states_train.pkl.gz', 'rb') as f:
        states = pickle.load(f)

    with gzip.open('../data/train/actions_train.pkl.gz', 'rb') as f:
        actions = pickle.load(f)

    with gzip.open('../data/train/distances_train.pkl.gz', 'rb') as f:
        distances = pickle.load(f)

    env = generate_env()

    r_s = np.asarray(room_structures)
    s = np.asarray(states)
    a = np.asarray(actions)
    d = np.asarray(distances)


    idx = randint(1, len(states))
    print('start game at index', idx)

    set_env_state(env, room_structures, states, idx)
    solve_game(env, actions, distances, idx)
