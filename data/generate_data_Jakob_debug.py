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



def drop_duplicate_states(states:list, room_structures:list, distances:list, actions:list):
    """
    Checks for duplicate states and chooses only those with the smallest provided distance.
    :return:
    """
    states, room_structures, distances, actions = np.asarray(states), np.asarray(room_structures)\
        , np.asarray(distances), np.asarray(actions)

    hashed_x = [[], []]  # [hash], [index in database] for states we want to keep
    hash_ind = 0
    for x in tqdm(states):
        x_hash = hash(marshal.dumps(x))
        try:
            duplicate = hashed_x[0].index(x_hash)  # attempt to find a duplicate
            duplicate_ind = hashed_x[1][duplicate]

            keep = duplicate_ind if distances[duplicate_ind] <= distances[hash_ind] else hash_ind

            hashed_x[1].append(keep)

        except ValueError:  # no duplicate found

            hashed_x[1].append(hash_ind)

        hashed_x[0].append(x_hash)
        hash_ind += 1

    keep = np.asarray(hashed_x[1])

    states, room_structures, distances, actions = states[keep], room_structures[keep], distances[keep], actions[keep]

    return list(states), list(room_structures), list(distances), list(actions)


def evaluate_and_save(states, room_structures, distances, actions, outfile_name):
    """
    Check for duplicates and save all games found so far (allows for generation and storage of games "in parallel")
    :return:
    """

    states, room_structures, distances, actions = drop_duplicate_states(states, room_structures, distances, actions)

    # save data
    print("Saving:", outfile_name)

    with gzip.open(f'./train/states_{outfile_name}.pkl.gz', 'wb') as f:
        pickle.dump(states, f, pickle.HIGHEST_PROTOCOL)

    with gzip.open(f'./train/distances_{outfile_name}.pkl.gz', 'wb') as f:
        pickle.dump(distances, f, pickle.HIGHEST_PROTOCOL)

    with gzip.open(f'./train/actions_{outfile_name}.pkl.gz', 'wb') as f:
        pickle.dump(actions, f, pickle.HIGHEST_PROTOCOL)

    with gzip.open(f'./train/room_structures_{outfile_name}.pkl.gz', 'wb') as f:
        pickle.dump(room_structures, f, pickle.HIGHEST_PROTOCOL)

    return states, room_structures, distances, actions



if __name__ == '__main__':
    env = SokobanEnv(dim_room=(10, 10), max_steps=200, num_boxes=3, num_gen_steps=None, reset=False)

    states, distances, actions, room_structures = [], [], [], []
    weird_states = []  # used for debugging
    timestamp = time.time()
    outfile_name = timestamp  # FIXME: Change this to add to existing database
    n_training_data = int(1e4)  # generate this many data
    save_every = 40  # save after this many games have been added

    for game_index in tqdm(range(n_training_data)):
        # for game_index in range(n_training_data):

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

                old_4 = state == 4
                old_3 = state == 3

                states.append(state)
                distances.append(distance)
                actions.append(action)
                room_structures.append(room_structure)

                state[old_4] = 3  # FIXME Here we swap the values to be like the game
                state[old_3] = 4  # FIXME

                # # FIXME debugging below
                # num_boxes_on_target = int(np.sum(state == 3))
                # total_boxes = int(np.sum((state == 3) | (state == 4)))
                # if distance == 1 and (num_boxes_on_target != 2 or total_boxes != 3):  # enforce the right number of boxes
                #     weird_states.append((state, distance, room_structure))  # error here
                #     pass
        else:
            print('🚨', 'could not solve a puzzle... skipping...')

        if (game_index + 1) % save_every == 0:

            states, room_structures, distances, actions = evaluate_and_save(states, room_structures, distances, actions,
                                                                            outfile_name)

    print('len(states)', len(states))

    # TODO remove duplicate state with bigger distance (Done: Jakob)

    evaluate_and_save(states, room_structures, distances, actions, outfile_name)
