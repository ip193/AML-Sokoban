import gym
import gym_sokoban
import time
import numpy as np


class Training:
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

        self.X, self.y = None, None  # FIXME initial states and the number of steps from a solution to generate them

        self.protocol = None  # this becomes a 2-tuple of lists
        # first list holds numbers of steps, second list holds number of training instances to look at at this distance

    def setData(self, X, y):  # FIXME
        self.X, self.y = X, y

    def setProtocol(self, steps, training_volume):
        self.protocol = (steps, training_volume)

    def initialize_to_state(self, env, state):  # FIXME
        """
        Set the game to a desired state
        :param env: Sokoban Environment whose state should be altered
        :param state: The new value for room_state
        :return:
        """

        env.reset()

        room_fixed = state.copy()

        room_fixed[np.where[room_fixed in (3, 4, 5)]] = 1

        env.room_state, env.room_fixed = state, room_fixed

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
            sample = np.argwhere(self.y == step_distance)  # FIXME

            for training_volume in self.protocol[1]:  # for this many episodes
                for tau in range(training_volume):
                    for ind, env in enumerate(envs):  # each agent
                        self.initialize_to_state(env, self.X[np.random.choice(sample)])

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
