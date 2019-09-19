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
from run_files.config import dtype, FILE_TRIES, SLEEP_TIME
from data.generate_data_Jakob_debug import load_data


class RetryFail:
    def __init__(self):
        self.ENV_FAILURE_RETRIES = 2
        self.failed_this_tau = False

    def env_fail(self, excp):
        if self.ENV_FAILURE_RETRIES > 0:
            print(excp)
            print("Warning: Environment failure encountered, retrying: ")
            self.ENV_FAILURE_RETRIES -= 1
            self.failed_this_tau = True
        else:
            raise excp


class TrainingToolsSokoban:
    """
    Hold training loop and info for Agent-like algorithms
    """

    def __init__(self, agents, grid_size=7, num_boxes=2, steps=100, save_every=100, reload_every=None,
                 test_every=50, game_name='Sokoban'):
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
        self.test_every = test_every

        self.game_name = game_name

        self.states, self.room_structures, self.distances = None, None, None  # Hold training data
        self.states_test, self.room_structures_test, self.distances_test = None, None, None

        self.protocol = None  # this becomes a 2-tuple of lists
        # first list holds numbers of steps, second list holds number of training instances to look at at this distance

        self.game_dict = {"Sokoban": SokobanEnv}

    def setData(self, filename):
        """
        Set file paths for data
        :param filename: Name of state, structures and distances files
        :param fileEnding: File ending (including leading period '.')
        :return:
        """
        self.data_filename = filename
        self.test_filename = filename+"-TEST"

    def loadData(self, retry=1, test=False):
        """
        Set the training data for this training cycle.
        Note: distances[idx  - t] holds number of steps left to solution from states[idx - t]

        :param filename: Name of state, structures and distances files
        :param fileEnding: File ending (including leading period '.')
        :return:
        """
        print("Setting data:", self.test_filename if test else self.data_filename)
        # filename = self.test_filename if test else self.data_filename
        # def f_open(names):
        #     for i in range(FILE_TRIES):
        #         ret = []
        #         try:
        #             for name in names:
        #                 if fileEnding == ".pkl.gz":
        #                     with gzip.open('../data/train/' + name + '_' + self.data_filename + self.fileEnding, 'rb') as f:
        #                         ret.append(pickle.load(f))
        #
        #                 elif fileEnding == ".npy":
        #                     ret.append(np.load('../data/train/' + name + '_' + self.data_filename + self.fileEnding))
        #
        #                 else:
        #                     raise RuntimeError("Invalid file ending received: "+fileEnding)
        #
        #             return tuple([np.asarray(v) for v in ret])
        #
        #         except Exception as e:
        #             print(e)
        #             print("Training data access failed. Retrying:")
        #             time.sleep(SLEEP_TIME)
        #
        #     raise IOError("Unable to access training data.")

        names = ("states", "room_structures", "distances")
        if not test:
            (self.states, self.room_structures, self.distances, actions) = load_data(self.data_filename, aslist=False,  # f_open(names)
                                                                                     folder='../data/train/')
        else:
            (self.states_test, self.room_structures_test, self.distances_test, actions_test) = load_data(self.test_filename, aslist=False,
                                                                                                         folder='../data/train/')
        try:
            assert(self.states.shape[0] == self.room_structures.shape[0] == self.distances.shape[0])
        except AssertionError as e:
            if retry == 0:
                raise e
            print("Warning: Reloaded database with faulty sizes. Retrying: ", retry)
            self.loadData(retry=retry-1, test=test)
        print("Database size:", self.states.shape[0])

    def setProtocol(self, steps:list, training_volume:list):
        """
        steps[i] holds distance from solution, training_volume[i] says how long to train at that distance.
        :param steps:
        :param training_volume:
        :return:
        """

        self.protocol = (np.asarray(steps), np.asarray(training_volume).astype(int))

    def initialize_to_state(self, env, index, test=False):
        """
        Set the game to a desired state. This is basically a re-implementation of env.reset()

        :param env: Sokoban Environment whose state should be altered
        :param index: Index in training database for state and room_structure
        :param test: If True, use index in testing database
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

    def refillEnvs(self, envs):
        """
        Reset the array of Sokoban environments.
        :param envs:
        :return:
        """
        envs.clear()
        for agent in self.agents:
            # envs.append(gym.make(env_name, dim_room=(self.grid_size, self.grid_size),
            #                     num_boxes=self.num_boxes))

            envs.append(self.game_dict[self.game_name](dim_room=(self.grid_size, self.grid_size)))


    def runTraining(self):
        """
        Execute the entire training process.
        :param reload: If not None, reload training data after this many games (allows for parallel data generation and loading)
        :return:
        """
        envs = []

        episodes = 0

        for ind_steps, step_distance in enumerate(self.protocol[0]):  # run games with this difficulty
            sample = np.where(self.distances == step_distance)[0]
            test_sample = np.where(self.distances_test == step_distance)[0]

            training_volume = self.protocol[1][ind_steps]

            self.refillEnvs(envs)
            print("Starting training at", step_distance, "steps from solution.")
            # for tau in tqdm(range(training_volume)):  # for this many episodes
            for tau in range(training_volume):
                env_fail = RetryFail()
                for ind_envs, env in enumerate(envs):  # each agent
                    try:
                        testing = False
                        if (episodes + 1) % self.test_every == 0:
                            start = np.random.choice(test_sample)
                            testing = True
                        else:
                            start = np.random.choice(sample)

                        self.initialize_to_state(env, start)

                        agent = self.agents[ind_envs]
                        agent.resetEpisode()

                        for t in range(step_distance*5):  # for this many steps
                            agent.giveObservation(self.getState(env))
                            action = agent.act(test=testing)
                            observation, reward, done, info = env.step(action)
                            agent.giveReward(reward)
                            agent.incrementTimeStep()

                            if done:
                                break

                        agent.endOfEpisodeUpdate(step_distance, test=testing)
                    except TypeError as e:
                        env_fail.env_fail(e)

                if env_fail.failed_this_tau:
                    self.refillEnvs(envs)

                episodes += 1

                if episodes % self.save_every == 0:
                    print("Saving agents, resetting envs.", episodes)
                    self.refillEnvs(envs)
                    # we clear the envs because env objects are buggy and sometimes fail
                    for agent in self.agents:
                        print("Total episodes for agent:", sum([len(agent.history.training_rewards[step_dist]) for step_dist in agent.history.training_rewards]))  # FIXME Doesn't work for non-SSRL
                        agent.save()
                if self.reload_every is not None and episodes % self.reload_every == 0:
                    print("Reloading data.", episodes)
                    self.loadData()
                    sample = np.where(self.distances == step_distance)[0]
                    test_sample = np.where(self.distances_test == step_distance)[0]

            print("Saving agents.")
            for agent in self.agents:
                agent.save()

class TrainingThreadSokoban(threading.Thread):
    """
    Runs training in its own thread (should pass agents list with only one element)
    """
    def __init__(self, agents, **kwargs): # steps=100, save_every=100, reload_every=None):
        super().__init__()
        self.training_tools = TrainingToolsSokoban(agents, **kwargs)  #  steps=steps, save_every=save_every, reload_every=reload_every)

    def run(self):
        print("Thread starting: ", self.training_tools.agents[0].name)
        self.training_tools.runTraining()

