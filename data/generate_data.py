import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import gzip
import marshal
import pickle
import time

import numpy as np
from gym_sokoban.envs import SokobanEnv
from gym_sokoban.envs.room_utils import room_topology_generation, box_displacement_score, ACTION_LOOKUP, reverse_move
from tqdm import tqdm

from run.example_fake_solved_game import generate_env, set_env_state, solve_game


def create_empty_room(env):
    return room_topology_generation(dim=env.dim_room, num_steps=env.num_gen_steps)


def place_boxes_and_player(room, num_boxes, second_player):
    # Get all available positions
    possible_positions = np.where(room == 1)
    num_possible_positions = possible_positions[0].shape[0]
    num_players = 2 if second_player else 1

    if num_possible_positions <= num_boxes + num_players:
        raise RuntimeError('Not enough free spots (#{}) to place {} player and {} boxes.'.format(
            num_possible_positions,
            num_players,
            num_boxes)
        )

    placed_player = False

    # place boxes
    for n in range(num_boxes):
        possible_positions = np.where(room == 1)
        num_possible_positions = possible_positions[0].shape[0]

        ind = np.random.randint(num_possible_positions)
        box_position = possible_positions[0][ind], possible_positions[1][ind]
        room[box_position] = 2

        # place player
        if not placed_player:
            for player_position in [
                (box_position[0] + 1, box_position[1]), (box_position[0] - 1, box_position[1]),
                (box_position[0], box_position[1] + 1), (box_position[0], box_position[1] - 1)
            ]:
                if room[player_position] == 1:
                    room[player_position] = 5
                    placed_player = True
                    break

    return room


def get_room_structure(room):
    # Room fixed represents all not movable parts of the room
    room_structure = np.copy(room)
    room_structure[room_structure == 5] = 1
    return room_structure


def get_room_state(room):
    # Room structure represents the current state of the room including movable parts
    room_state = room.copy()
    room_state[room_state == 2] = 4
    return room_state


def get_box_mapping(room_structure):
    # Box_Mapping is used to calculate the box displacement for every box
    box_mapping = {}
    box_locations = np.where(room_structure == 2)
    num_boxes = len(box_locations[0])
    for l in range(num_boxes):
        box = (box_locations[0][l], box_locations[1][l])
        box_mapping[box] = box
    return box_mapping


# Global variables used for reverse playing.
explored_states = set()
num_boxes = 0
best_room_score = -1
best_room = None
best_box_mapping = None
best_actions = None
best_old_room_states = None
best_distances = None


def reset_globals():
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping, best_actions, best_old_room_states, best_distances
    explored_states = set()
    num_boxes = 0
    best_room_score = -1
    best_room = None
    best_box_mapping = None
    best_actions = None
    best_old_room_states = None
    best_distances = None


def reverse_playing(room_state, room_structure, search_depth=100):
    """
    This function plays Sokoban reverse in a way, such that the player can
    move and pull boxes.
    It ensures a solvable level with all boxes not being placed on a box target.
    :param room_state:
    :param room_structure:
    :param search_depth:
    :return: 2d array
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping

    reset_globals()

    # Box_Mapping is used to calculate the box displacement for every box
    box_mapping = {}
    box_locations = np.where(room_structure == 2)
    num_boxes = len(box_locations[0])
    for l in range(num_boxes):
        box = (box_locations[0][l], box_locations[1][l])
        box_mapping[box] = box

    # explored_states globally stores the best room state and score found during search
    explored_states = set()
    best_room_score = -1
    best_box_mapping = box_mapping
    depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300)

    return best_room, best_room_score, best_box_mapping


def depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300, actions=None, old_room_states=None, distances=None, max_action_length=30):
    """
    Searches through all possible states of the room.
    """
    if actions is None:
        actions = [-1]
    if old_room_states is None:
        old_room_states = []
    if distances is None:
        distances = []

    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping, best_actions, best_old_room_states, best_distances

    ttl -= 1
    if ttl <= 0 or len(explored_states) >= 100000:
        return

    state_tohash = marshal.dumps(room_state)

    # Only search this state, if it not yet has been explored
    if not (state_tohash in explored_states):

        if len(distances) == 0:
            distances.append(0)
            old_room_states.append(room_state)

        # Add current state and its score to explored states
        room_score = box_swaps * box_displacement_score(box_mapping)
        if np.where(room_state == 2)[0].shape[0] != num_boxes:
            room_score = 0

        if room_score > best_room_score:
            best_room = room_state
            best_room_score = room_score
            best_box_mapping = box_mapping
            best_actions = actions
            best_old_room_states = old_room_states
            best_distances = distances

        explored_states.add(state_tohash)

        for action in ACTION_LOOKUP.keys():
            # The state and box mapping  need to be copied to ensure
            # every action start from a similar state.
            room_state_next = room_state.copy()
            box_mapping_next = box_mapping.copy()
            actions_next = actions.copy()
            old_room_states_next = old_room_states.copy()
            distances_next = distances.copy()

            room_state_next, box_mapping_next, last_pull_next = \
                reverse_move(room_state_next, room_structure, box_mapping_next, last_pull, action)

            box_swaps_next = box_swaps
            if not np.array_equal(room_state_next, room_state):  # only use room_state if something has changed
                if len(np.where(room_state_next == 2)[0]) > 0:  # only save actions if not solved
                    actions_next.append(action)
                    distances_next.append(len(actions_next) - 1)  # last action is nop
                    old_room_states_next.append(room_state_next)

                if last_pull_next != last_pull:
                    box_swaps_next += 1

            if len(actions_next) < max_action_length:
                depth_first_search(room_state_next, room_structure,
                                   box_mapping_next, box_swaps_next,
                                   last_pull, ttl, actions_next, old_room_states_next, distances_next)


def action_solver(actions):
    action_mapper = {-1: -1, 0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}
    solution = []
    for action in actions:
        solution.append(action_mapper[action])
    return solution


def solution_works(room_structures, room_states, actions, distances, idx):
    env = generate_env()
    set_env_state(env, room_structures, room_states, idx)
    return solve_game(env, actions, distances, idx, render_mode='tiny_rgb_array')


if __name__ == '__main__':
    env = SokobanEnv(dim_room=(10, 10), max_steps=200, num_boxes=3, num_gen_steps=None, reset=False)

    states, distances, actions, room_structures = [], [], [], []
    for _ in tqdm(range(1000)):

        room_structure = None
        score = 0
        while score == 0:
            try:
                room = create_empty_room(env)
                room = place_boxes_and_player(room, num_boxes=env.num_boxes, second_player=False)

                room_structure = get_room_structure(room)
                room_state = get_room_state(room)

                room_state, score, box_mapping = reverse_playing(room_state, room_structure)

                env.room_fixed, env.room_state, env.box_mapping = room, room_state, get_box_mapping(room_structure)
            except:
                pass

        actions_solution = action_solver(best_actions)

        if solution_works([room_structure] * len(best_old_room_states), best_old_room_states, actions_solution, best_distances, len(best_old_room_states) - 1):
            for state, distance, action in zip(best_old_room_states, best_distances, actions_solution):
                states.append(state)
                distances.append(distance)
                actions.append(action)
                room_structures.append(room_structure)
        else:
            print('🚨', 'could not solve a puzzle... skipping...')

    print('len(states)', len(states))

    # TODO remove duplicate state with bigger distance
    hashed_x = []
    for x in tqdm(states.copy()):
        x_hash = hash(marshal.dumps(x))
        if x_hash in hashed_x:
            print("! Duplicate in Dataset!!!")
        else:
            hashed_x.append(x_hash)

    # save data
    timestamp = time.time()
    print(timestamp)
    with gzip.open(f'states_{timestamp}.pkl.gz', 'wb') as f:
        pickle.dump(states, f, pickle.HIGHEST_PROTOCOL)

    with gzip.open(f'distances_{timestamp}.pkl.gz', 'wb') as f:
        pickle.dump(distances, f, pickle.HIGHEST_PROTOCOL)

    with gzip.open(f'actions_{timestamp}.pkl.gz', 'wb') as f:
        pickle.dump(actions, f, pickle.HIGHEST_PROTOCOL)

    with gzip.open(f'room_structures_{timestamp}.pkl.gz', 'wb') as f:
        pickle.dump(room_structures, f, pickle.HIGHEST_PROTOCOL)
