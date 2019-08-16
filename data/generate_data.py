import time

import numpy as np
from gym_sokoban.envs import SokobanEnv
from gym_sokoban.envs.room_utils import room_topology_generation


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


if __name__ == '__main__':
    env = SokobanEnv(dim_room=(7, 7), max_steps=200, num_boxes=2, num_gen_steps=None, reset=False)
    room = create_empty_room(env)
    room = place_boxes_and_player(room, num_boxes=env.num_boxes, second_player=False)

    room_structure = get_room_structure(room)
    room_state = get_room_state(room)
    env.room_fixed, env.room_state, env.box_mapping = room, room_state, get_box_mapping(room_structure)

    # TODO move player and boxes
    # TODO record movements of player (state + action)
    # TODO save data

    # visualize
    env.render('human')
    time.sleep(10)
