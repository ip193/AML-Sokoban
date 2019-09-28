import gc
import hashlib
import marshal
from copy import deepcopy
from heapq import *

import gym
import gym_sokoban
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym_sokoban.envs import ACTION_LOOKUP
from scipy.special._ufuncs import binom
from torch import nn
from tqdm import tqdm

gym_sokoban  # PyCharm hack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(state_dict_path, input_size=7 * 7 * 4):
    def create_model():
        class ResidualBlock(nn.Module):
            expansion = 1

            def __init__(self, num_features):
                super(ResidualBlock, self).__init__()

                self.fc1 = nn.Linear(num_features, num_features)
                # self.bn1 = nn.BatchNorm1d(num_features)
                self.relu = nn.ReLU(inplace=True)
                self.fc2 = nn.Linear(num_features, num_features)
                # self.bn2 = nn.BatchNorm1d(num_features)

            def forward(self, x):
                identity = x

                out = self.fc1(x)
                # out = self.bn1(out)
                out = self.relu(out)

                out = self.fc2(out)
                # out = self.bn2(out)

                out += identity

                return self.relu(out)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.model = nn.Sequential(
                    nn.Linear(input_size, 5000),
                    # nn.BatchNorm1d(5000),
                    nn.ReLU(),

                    nn.Linear(5000, 1000),
                    # nn.BatchNorm1d(1000),
                    nn.ReLU(),

                    ResidualBlock(1000),
                    ResidualBlock(1000),
                    ResidualBlock(1000),
                    ResidualBlock(1000),

                    nn.Linear(1000, 1)
                )

            def forward(self, x):
                return self.model(x)

        return Net()

    model = create_model().to(DEVICE)
    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device(DEVICE)))
    model.eval()
    return model


def heuristic(env, model):
    room_state, room_structure = env.room_state, env.room_fixed
    wall_map = torch.from_numpy((room_state == 0).astype(int)).flatten()
    target_map = torch.from_numpy((room_structure == 2).astype(int)).flatten()
    boxes_map = torch.from_numpy(((room_state == 3) | (room_state == 4)).astype(int)).flatten()
    agent_map = torch.from_numpy((room_state == 5).astype(int)).flatten()

    state = torch.cat((wall_map, target_map, boxes_map, agent_map), 0)
    state = state.reshape(1, -1)

    with torch.no_grad():
        predict = model(state.float().to(DEVICE))

    return predict.detach().cpu().numpy()[0][0]


def search_way(start_env, model, epsilon=0.000000001, weight=1, progress_bar_update_iterations=10):
    x = sum((start_env.room_state > 0).flatten())
    progress_bar = tqdm(total=int(x * binom(x - 1, start_env.num_boxes)))

    start_env_md5 = hashlib.md5(marshal.dumps(start_env.room_state)).hexdigest()

    close_set = set()
    came_from = {}
    gscore = {start_env_md5: 0}
    fscore = {start_env_md5: heuristic(start_env, model)}
    open_heap = []
    open_heap_md5_dict = {}
    open_heap_fscore_set = set()
    actions = {}

    def add_to_open_heap_md5_dict(fscore, key):
        if key in open_heap_md5_dict:
            open_heap_md5_dict[key] += 1
        else:
            open_heap_md5_dict[key] = 1

        open_heap_fscore_set.add(fscore)

    def remove_from_open_heap_md5_dict(fscore, key):
        open_heap_md5_dict[key] -= 1
        if open_heap_md5_dict[key] == 0:
            del open_heap_md5_dict[key]

        if fscore in open_heap_fscore_set:
            open_heap_fscore_set.remove(fscore)

    heappush(open_heap, (fscore[start_env_md5], [start_env.room_state, start_env.room_fixed, False]))
    add_to_open_heap_md5_dict(fscore[start_env_md5], start_env_md5)

    counter = 0
    current = deepcopy(start_env)
    current.set_maxsteps(999999)
    while open_heap:
        current.room_state, current.room_fixed, current.done = heappop(open_heap)[1]  # pop the smallest item off the heap
        player_position = np.where(current.room_state == 5)
        if len(player_position[0]) > 1:
            print('ERROR', player_position)
        current.player_position = np.array([player_position[0][0], player_position[1][0]])

        current_md5 = hashlib.md5(marshal.dumps(current.room_state)).hexdigest()
        remove_from_open_heap_md5_dict(fscore[current_md5], current_md5)

        counter += 1
        if counter % progress_bar_update_iterations == 0:
            progress_bar.update(progress_bar_update_iterations)
            if counter % 10000 == 0:
                gc.collect()

        if current.done:
            steps = []
            while current_md5 in came_from:
                steps.append(actions[current_md5])
                current_md5 = came_from[current_md5]

            progress_bar.update(counter % progress_bar_update_iterations)
            progress_bar.close()
            return list(reversed(steps)), len(close_set)

        close_set.add(current_md5)
        for action in [1, 2, 3, 4]:  # skip 0 as NOP
            tentative_g_score = gscore[current_md5] + 1

            neighbor_env = deepcopy(current)
            _, _, neighbor_env.done, _ = neighbor_env.step(action)

            neighbor_env_md5 = hashlib.md5(marshal.dumps(neighbor_env.room_state)).hexdigest()

            if np.array_equal(current.room_state, neighbor_env.room_state):  # nothing changed
                continue

            if neighbor_env_md5 in close_set and tentative_g_score >= gscore.get(neighbor_env_md5, 0):
                continue

            if tentative_g_score < gscore.get(neighbor_env_md5, 0) or neighbor_env_md5 not in open_heap_md5_dict:
                came_from[neighbor_env_md5] = current_md5
                gscore[neighbor_env_md5] = tentative_g_score
                fscore[neighbor_env_md5] = tentative_g_score * weight + heuristic(neighbor_env, model)
                actions[neighbor_env_md5] = action
                while fscore[neighbor_env_md5] in open_heap_fscore_set:
                    fscore[neighbor_env_md5] += epsilon
                heappush(open_heap, (fscore[neighbor_env_md5], [neighbor_env.room_state, neighbor_env.room_fixed, neighbor_env.done]))
                add_to_open_heap_md5_dict(fscore[neighbor_env_md5], neighbor_env_md5)

    progress_bar.close()
    return False, False


def heuristic_parallel(envs, model):
    states = torch.from_numpy(np.asarray([])).flatten()
    for env in envs:
        wall_map = torch.from_numpy((env.room_state == 0).astype(int)).flatten()
        target_map = torch.from_numpy((env.room_fixed == 2).astype(int)).flatten()
        boxes_map = torch.from_numpy(((env.room_state == 3) | (env.room_state == 4)).astype(int)).flatten()
        agent_map = torch.from_numpy((env.room_state == 5).astype(int)).flatten()

        state = torch.cat((wall_map, target_map, boxes_map, agent_map), 0)
        states = torch.cat((states.float(), state.float()), 0)
    states = states.reshape(len(envs), -1)

    with torch.no_grad():
        predict = model(states.float().to(DEVICE))

    return predict.detach().cpu().numpy().flatten()


def search_way_parallel(start_env, model, epsilon=0.000000001, weight=1, progress_bar_update_iterations=10):
    x = sum((start_env.room_state > 0).flatten())
    progress_bar = tqdm(total=int(x * binom(x - 1, start_env.num_boxes)))

    start_env_md5 = hashlib.md5(marshal.dumps(start_env.room_state)).hexdigest()

    close_set = set()
    came_from = {}
    gscore = {start_env_md5: 0}
    fscore = {start_env_md5: heuristic_parallel([start_env], model)[0]}
    open_heap = []
    open_heap_md5_dict = {}
    open_heap_fscore_set = set()
    actions = {}

    def add_to_open_heap_md5_dict(fscore, key):
        if key in open_heap_md5_dict:
            open_heap_md5_dict[key] += 1
        else:
            open_heap_md5_dict[key] = 1

        open_heap_fscore_set.add(fscore)

    def remove_from_open_heap_md5_dict(fscore, key):
        open_heap_md5_dict[key] -= 1
        if open_heap_md5_dict[key] == 0:
            del open_heap_md5_dict[key]

        if fscore in open_heap_fscore_set:
            open_heap_fscore_set.remove(fscore)

    heappush(open_heap, (fscore[start_env_md5], [start_env.room_state, start_env.room_fixed, False]))
    add_to_open_heap_md5_dict(fscore[start_env_md5], start_env_md5)

    counter = 0
    current = deepcopy(start_env)
    current.set_maxsteps(999999)
    while open_heap:
        current.room_state, current.room_fixed, current.done = heappop(open_heap)[1]  # pop the smallest item off the heap
        player_position = np.where(current.room_state == 5)
        if len(player_position[0]) > 1:
            print('ERROR', player_position)
        current.player_position = np.array([player_position[0][0], player_position[1][0]])

        current_md5 = hashlib.md5(marshal.dumps(current.room_state)).hexdigest()
        remove_from_open_heap_md5_dict(fscore[current_md5], current_md5)

        counter += 1
        if counter % progress_bar_update_iterations == 0:
            progress_bar.update(progress_bar_update_iterations)
            if counter % 10000 == 0:
                gc.collect()

        if current.done:
            steps = []
            while current_md5 in came_from:
                steps.append(actions[current_md5])
                current_md5 = came_from[current_md5]

            progress_bar.update(counter % progress_bar_update_iterations)
            progress_bar.close()
            return list(reversed(steps)), len(close_set)

        close_set.add(current_md5)
        temp_parallel = []
        for action in [1, 2, 3, 4]:  # skip 0 as NOP
            tentative_g_score = gscore[current_md5] + 1

            neighbor_env = deepcopy(current)
            _, _, neighbor_env.done, _ = neighbor_env.step(action)

            neighbor_env_md5 = hashlib.md5(marshal.dumps(neighbor_env.room_state)).hexdigest()

            if np.array_equal(current.room_state, neighbor_env.room_state):  # nothing changed
                continue

            if neighbor_env_md5 in close_set and tentative_g_score >= gscore.get(neighbor_env_md5, 0):
                continue

            if tentative_g_score < gscore.get(neighbor_env_md5, 0) or neighbor_env_md5 not in open_heap_md5_dict:
                came_from[neighbor_env_md5] = current_md5
                gscore[neighbor_env_md5] = tentative_g_score
                actions[neighbor_env_md5] = action

                temp_parallel.append([neighbor_env_md5, neighbor_env])

        if len(temp_parallel) > 0:
            for heuristic_result, (neighbor_env_md5, neighbor_env) in zip(heuristic_parallel([neighbor_env for _, neighbor_env in temp_parallel], model), temp_parallel):
                fscore[neighbor_env_md5] = gscore[neighbor_env_md5] * weight + heuristic_result
                while fscore[neighbor_env_md5] in open_heap_fscore_set:
                    fscore[neighbor_env_md5] += epsilon
                heappush(open_heap, (fscore[neighbor_env_md5], [neighbor_env.room_state, neighbor_env.room_fixed, neighbor_env.done]))
                add_to_open_heap_md5_dict(fscore[neighbor_env_md5], neighbor_env_md5)

    progress_bar.close()
    return False, False


if __name__ == '__main__':
    model = load_model('DAVI_steps_10_longTraining_no_batchnorm.pth')
    env = gym.make('Sokoban-small-v0')
    plt.imshow(env.render('rgb_array'))
    plt.show()

    result, explored_len = search_way(env, model)
    if type(result) is list and len(result) > 0:
        print(f'total of {len(result)} steps ({len(result) / float(explored_len) * 100:0.5f}% of explored steps)')
        x = sum((env.room_state > 0).flatten())
        print(f'explored {len(result) / float(x * binom(x - 1, env.num_boxes)) * 100:0.5f}% of all possible states')
        for action_idx in result:
            print(ACTION_LOOKUP[action_idx])
    else:
        print('ERROR, could not find a path 😢')
