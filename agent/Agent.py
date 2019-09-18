
from abc import ABCMeta, abstractmethod
import pickle
from time import sleep

from run_files.config import FILE_TRIES, SLEEP_TIME

class SaveInfo:
    """
    Holds information about where to save objects and logs, e.g. folder names,
    """

    def __init__(self, agent, special_name_tag=None):

        self.agent = agent
        self.filename = self.agent.name if special_name_tag is None else \
            self.agent.name + "_" + special_name_tag

        self.dir = "../data/models"

    def save(self):
        done = [False]
        for i in range(FILE_TRIES):
            try:
                with open(self.dir + "/" + self.filename+".pkl", "wb+") as pickle_out:
                    pickle.dump(self.agent, pickle_out)
                done[0] = True
            except:
                sleep(SLEEP_TIME)
                pass
            if done[0]:
                break

    def load(self):
        for i in range(FILE_TRIES):
            try:
                with open(self.dir + "/" + self.filename+".pkl", "rb") as pickle_in:
                    return pickle.load(pickle_in)
            except:
                print("Agent loading failed, retrying:", self.agent.name)
                sleep(SLEEP_TIME)
                pass
        raise FileNotFoundError("Agent not found.")


class Agent(metaclass=ABCMeta):
    """
    Agent interface to handle action output, rewards, etc.
    """

    def __init__(self):
        self.name = None
        self.save_info = None
        pass

    @abstractmethod
    def resetEpisode(self, episode_tag):
        """
        Clear buffers/running values, rename episode, and reset time step to 0.
        :param episode_tag: New episode name
        :return:
        """
        pass

    @abstractmethod
    def incrementTimeStep(self):
        """
        Increase the time step in this episode by one.
        :return:
        """
        pass

    @abstractmethod
    def giveObservation(self, obs):
        """
        Provide the observation to the agent.
        :param obs: Observation of the world (numpy array)
        :return:
        """
        pass

    @abstractmethod
    def act(self):
        """
        Return the action label (after having received the observation).
        :return: Action label
        """
        pass

    @abstractmethod
    def giveReward(self, reward):
        """
        Provide the reward to the agent for the last action taken.
        :return:
        """
        pass

    @abstractmethod
    def endOfEpisodeUpdate(self, **kwargs):
        """
        Update parameters after the conclusion of an episode.
        :return:
        """
        pass

    def setSaveInfo(self, special_name_tag=None):
        self.save_info = SaveInfo(self, special_name_tag=special_name_tag)

    def save(self):
        self.save_info.save()

    def load(self):
        return self.save_info.load()
