import gym
import gym_sokoban
import time

# Before you can make a Sokoban Environment you need to call:
# import gym_sokoban
# This import statement registers all Sokoban environments
# provided by this package


class SaveInfo:
    """
    Holds information about where to save objects and logs
    """

    def __init__(self):

        pass
        # TODO



class Training:
    """
    Hold training loop and info for Agent-like algorithms
    """

    def __init__(self, agents, save_every=100):
        """

        :param agents: List of agents to train
        :param save_every: Save every n episodes
        """
        self.agents = agents
        self.save_info = None
        self.save_every = save_every

        self.X, self.y = None, None  # initial states and the number of steps from a solution to generate them

        self.protocol = None  # this becomes a 2-tuple of lists
        # first list holds numbers of steps, second list holds number of training instances to look at at this distance

    def setSaveInfo(self, save):
        self.save_info = save

    def setData(self, X, y):
        self.X, self.y = X, y

    def setProtocol(self, steps, training_volume):
        self.protocol = (steps, training_volume)

    def initialize_to_state(env, state):
        """
        Set the game to a desired state
        :param env:
        :param state:
        :return:
        """

        # TODO

        ...

    def runTraining(self):
        """
        Execute the entire training process.
        :return:
        """

        # TODO
        ...


env_name = 'Sokoban-v0'
env = gym.make(env_name)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))



for i_episode in range(1):#20
    observation = env.reset()

    for t in range(100):#100
        env.render(mode='human')
        action = env.action_space.sample()

        # Sleep makes the actions visible for users
        time.sleep(1)
        observation, reward, done, info = env.step(action)

        print(ACTION_LOOKUP[action], reward, done, info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.render()
            break

    env.close()

time.sleep(10)