import gym
import gym_sokoban
import time
import numpy as np
import pickle
import gzip
import os
from tqdm import tqdm
import threading
from gym_sokoban.envs import SokobanEnv
from data.generate_data import get_box_mapping
from agent.Agent import FILE_TRIES, SLEEP_TIME
from data.generate_data_Jakob_debug import load_data

class TrainingTools:
    """
    Hold training loop and info for Agent-like algorithms
    """

    def __init__(self, agents, grid_size=7, num_boxes=2, steps=100, save_every=100, reload_every=None):
        """

        :param steps: How many steps per episode
        :param agents: List of agents to train
        :param save_every: Save every n episodes
        """

        self.steps = steps
        self.grid_size = grid_size
        self.num_boxes = num_boxes

        self.agents = agents
        self.save_every = save_every
        self.reload_every = reload_every

        self.states, self.room_structures, self.distances = None, None, None  # Hold training data

        self.protocol = None  # this becomes a 2-tuple of lists
        # first list holds numbers of steps, second list holds number of training instances to look at at this distance

    def setData(self, filename, fileEnding=".pkl.gz", retry=1):
        """
        Set the training data for this training cycle.
        Note: distances[idx  - t] holds number of steps left to solution from states[idx - t]

        :param filename: Name of state, structures and distances files
        :param fileEnding: File ending (including leading period '.')
        :return:
        """
        self.data_filename = filename
        self.data_fileEnding = fileEnding

        print("Setting training data:", self.data_filename)
        def f_open(names):
            for i in range(FILE_TRIES):
                ret = []
                try:
                    for name in names:
                        if fileEnding == ".pkl.gz":
                            with gzip.open('../data/train/' + name + '_' + self.data_filename + self.data_fileEnding, 'rb') as f:
                                ret.append(pickle.load(f))

                        elif fileEnding == ".npy":
                            ret.append(np.load('../data/train/' + name + '_' + self.data_filename + self.data_fileEnding))

                        else:
                            raise RuntimeError("Invalid file ending received: "+fileEnding)

                    return tuple([np.asarray(v) for v in ret])

                except Exception as e:
                    print(e)
                    print("Training data access failed. Retrying:")
                    time.sleep(SLEEP_TIME)

            raise IOError("Unable to access training data.")

        names = ("states", "room_structures", "distances")
        (self.states, self.room_structures, self.distances, actions) = load_data(self.data_filename, aslist=False,  # f_open(names)
                                                                                 folder='../data/train/')
        try:
            assert(self.states.shape[0] == self.room_structures.shape[0] == self.distances.shape[0])
        except AssertionError as e:
            if retry == 0:
                raise e
            print("Warning: Reloaded database with faulty sizes. Retrying: ", retry)
            self.setData(filename, fileEnding, retry=retry-1)
        print("Database size:", self.states.shape[0])

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
        Set the game to a desired state. This is basically a re-implementation of env.reset()

        :param env: Sokoban Environment whose state should be altered
        :param index: Index in training database for state and room_structure
        :return:
        """

        #  env.reset()
        env.room_fixed = self.room_structures[index]
        env.room_state = self.states[index]
        env.box_mapping = get_box_mapping(self.room_structures[index])

        player_position = np.where(env.room_state == 5)
        env.player_position = np.asarray([player_position[0][0], player_position[1][0]])

        env.num_env_steps = 0
        env.reward_last = 0
        env.boxes_on_target = int(np.sum(env.room_state == 3))


    def getState(self, env):
        """
        Access the room's state (1: free space, 2: goal, 3: box on goal, 4: box free, 5: player)
        :param env:
        :return:
        """

        return env.room_state

    def runTraining(self):
        """
        Execute the entire training process.
        :param reload: If not None, reload training data after this many games (allows for parallel data generation and loading)
        :return:
        """

        env_name = 'Sokoban-v0'
        envs = []

        episodes = 0

        for ind_steps, step_distance in enumerate(self.protocol[0]):  # run games with this difficulty
            sample = np.where(self.distances == step_distance)[0]

            training_volume = self.protocol[1][ind_steps]

            envs.clear()
            for agent in self.agents:

                # envs.append(gym.make(env_name, dim_room=(self.grid_size, self.grid_size),
                #                     num_boxes=self.num_boxes))

                envs.append(SokobanEnv(dim_room=(self.grid_size, self.grid_size),
                                     num_boxes=self.num_boxes))  # FIXME
            print("Starting training at", step_distance, "steps from solution.")
            # for tau in tqdm(range(training_volume)):  # for this many episodes
            for tau in range(training_volume):
                for ind_envs, env in enumerate(envs):  # each agent
                    start = np.random.choice(sample)
                    self.initialize_to_state(env, start)

                    agent = self.agents[ind_envs]
                    agent.resetEpisode()

                    #FIXME debug
                    agent.start = self.states[start]
                    agent.actions = []  # holds actions taken this game

                    for t in range(step_distance*5):  # for this many steps
                        agent.giveObservation(self.getState(env))
                        action = agent.act()
                        agent.actions.append(action)  # FIXME
                        observation, reward, done, info = env.step(action)
                        agent.giveReward(reward)
                        agent.incrementTimeStep()

                        if done:
                            break

                    agent.endOfEpisodeUpdate(step_distance)

                episodes += 1

                if episodes % self.save_every == 0:
                    print("Saving agents.", episodes)
                    for agent in self.agents:
                        print("Total episodes for agent:", sum([len(agent.history.past_rewards[step_dist]) for step_dist in agent.history.past_rewards]))  # FIXME Doesn't work for non-SSRL
                        agent.save()
                if self.reload_every is not None and episodes % self.reload_every == 0:
                    print("Reloading data.", episodes)
                    self.setData(self.data_filename, self.data_fileEnding)
                    sample = np.where(self.distances == step_distance)[0]

            print("Saving agents.")
            for agent in self.agents:
                agent.save()

class TrainingThread(threading.Thread):
    """
    Runs training in its own thread (should pass agents list with only one element)
    """
    def __init__(self, agents, **kwargs): # steps=100, save_every=100, reload_every=None):
        super().__init__()
        self.training_tools = TrainingTools(agents, **kwargs)  #  steps=steps, save_every=save_every, reload_every=reload_every)

    def run(self):
        print("Thread starting: ", self.training_tools.agents[0].name)
        self.training_tools.runTraining()

