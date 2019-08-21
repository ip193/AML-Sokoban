import gym
import gym_sokoban
import time
import numpy as np
import pickle
import gzip
import os
from tqdm import tqdm


class TrainingTools:
    """
    Hold training loop and info for Agent-like algorithms
    """

    def __init__(self, agents, steps=100, save_every=100):
        """

        :param steps: How many steps per episode
        :param agents: List of agents to train
        :param save_every: Save every n episodes
        """

        self.steps = steps

        self.agents = agents
        self.save_every = save_every

        self.states, self.room_structures, self.distances = None, None, None  # Hold training data

        self.protocol = None  # this becomes a 2-tuple of lists
        # first list holds numbers of steps, second list holds number of training instances to look at at this distance

    def setData(self, filename, fileEnding=".pkl.gz"):
        """
        Set the training data for this training cycle.
        Note: distances[idx  - t] holds number of steps left to solution from states[idx - t]

        :param filename: Name of state, structures and distances files
        :param fileEnding: File ending (including leading period '.')
        :return:
        """

        def f_open(name):
            if fileEnding == ".pkl.gz":
                with gzip.open('../data/train/' + name + '_' + filename + fileEnding, 'rb') as f:
                    return pickle.load(f)

            if fileEnding == ".npy":
                return np.load('../data/train/' + name + '_' + filename + fileEnding)

            else:
                raise RuntimeError("Invalid file ending received: "+fileEnding)

        self.states, self.room_structures, self.distances = f_open("states"), f_open("room_structures"), \
                                                            f_open("distances")

    def setProtocol(self, steps:list, training_volume:list):
        """
        steps[i] holds distance from solution, training_volume[i] says how long to train at that distance.
        :param steps:
        :param training_volume:
        :return:
        """

        self.protocol = (np.asarray(steps), np.asarray(training_volume).astype(int))

    def initialize_to_state(self, env, index):
        """
        Set the game to a desired state.

        :param env: Sokoban Environment whose state should be altered
        :param index: Index in training database for state and room_structure
        :return:
        """

        env.reset()
        env.room_fixed = self.room_structures[index]
        env.room_state = self.states[index]

        old_4 = env.room_state == 4
        old_3 = env.room_state == 3

        env.room_state[old_4] = 3  # FIXME
        env.room_state[old_3] = 4  # FIXME

        player_position = np.where(env.room_state == 5)
        env.player_position = np.asarray([player_position[0][0], player_position[1][0]])

    def getState(self, env):
        """
        Access the room's state
        :param env:
        :return:
        """

        return env.room_state

    def runTraining(self):
        """
        Execute the entire training process.
        :return:
        """

        env_name = 'Sokoban-v0'
        envs = []
        for agent in self.agents:
            envs.append(gym.make(env_name))

        # ACTION_LOOKUP = envs[0].unwrapped.get_action_lookup()
        episodes = 0

        for step_distance in self.protocol[0]:  # run games with this difficulty
            sample = self.states[np.where(self.distances == step_distance)]  # FIXME

            for training_volume in self.protocol[1]:  # for this many episodes
                print("Starting training at", training_volume, "steps from solution.")
                for tau in tqdm(range(training_volume)):
                    for ind, env in enumerate(envs):  # each agent
                        self.initialize_to_state(env, np.random.choice(sample))

                        agent = self.agents[ind]

                        agent.resetEpisode()

                        for t in range(self.steps):  # for this many steps
                            agent.giveObservation(self.getState(env))
                            action = agent.act()
                            observation, reward, done, info = env.step(action)
                            agent.giveReward(reward)

                            if done:
                                break

                        agent.endOfEpisodeUpdate()

                    episodes += 1

                    if episodes % self.save_every == 0:
                        for agent in self.agents:
                            agent.save()  # TODO
