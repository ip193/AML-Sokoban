import marshal
from copy import deepcopy
from heapq import *

import gym
import gym_sokoban
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym_sokoban.envs import ACTION_LOOKUP
from torch import nn

gym_sokoban  # PyCharm hack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(state_dict_path):
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
                    nn.Linear(7 * 7 * 4, 5000),
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


def search_way(start_env, model, epsilon=0.000000001):
    close_set = set()
    came_from = {}
    gscore = {start_env: 0}
    fscore = {start_env: heuristic(start_env, model)}
    open_heap = []
    actions = {}

    heappush(open_heap, (fscore[start_env], start_env))

    while open_heap:
        current = heappop(open_heap)[1]  # pop the smallest item off the heap

        if getattr(current, 'done', False):
            steps = []
            while current in came_from:
                steps.extend(actions[current])
                current = came_from[current]
            return list(reversed(steps)), len(close_set)

        close_set.add(marshal.dumps(current.room_state))
        for action in [1, 2, 3, 4]:  # skip 0 as NOP
            tentative_g_score = gscore[current] + 1

            neighbor_env = deepcopy(current)
            _, _, done, _ = neighbor_env.step(action)
            neighbor_env.done = done

            if np.array_equal(current.room_state, neighbor_env.room_state):  # nothing changed
                continue

            if marshal.dumps(neighbor_env.room_state) in close_set and tentative_g_score >= gscore.get(neighbor_env, 0):
                continue

            if tentative_g_score < gscore.get(neighbor_env, 0) or marshal.dumps(neighbor_env.room_state) not in [marshal.dumps(i[1].room_state) for i in open_heap]:
                came_from[neighbor_env] = current
                gscore[neighbor_env] = tentative_g_score
                fscore[neighbor_env] = tentative_g_score + heuristic(neighbor_env, model)
                if neighbor_env not in actions:
                    actions[neighbor_env] = []
                actions[neighbor_env].append(action)
                while fscore[neighbor_env] in [i[0] for i in open_heap]:
                    fscore[neighbor_env] += epsilon
                heappush(open_heap, (fscore[neighbor_env], neighbor_env))

    return False


if __name__ == '__main__':
    model = load_model('DAVI_steps_10_longTraining_no_batchnorm.pth')
    env = gym.make('Sokoban-small-v0')
    plt.imshow(env.render('rgb_array'))
    plt.show()

    result, explored_len = search_way(env, model)
    if type(result) is list and len(result) > 0:
        print(f'total of {len(result)} steps ({len(result) / float(explored_len) * 100:0.5f}% of explored steps)')
        print(f'explored {len(result) / float((sum((env.room_state > 0).flatten())-3)**3) * 100:0.5f}% of all possible states')
        for action_idx in result:
            print(ACTION_LOOKUP[action_idx])
    else:
        print('ERROR, could not find a path 😢')
