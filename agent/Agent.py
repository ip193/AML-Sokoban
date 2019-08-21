
from abc import ABCMeta, abstractmethod
import numpy as np
import pickle


class SaveInfo:
    """
    Holds information about where to save objects and logs, e.g. folder names,
    """

    def __init__(self, agent, special_name_tag=None):

        self.agent = agent
        self.filename = self.agent.name if special_name_tag is not None else \
            self.agent.name + special_name_tag

        self.dir = "..data/models"

    def save(self):
        pickle_out = open(self.dir + "/" + self.filename+".pkl", "wb")
        pickle.dump(self.agent, pickle_out)
        pickle_out.close()


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
    def endOfEpisodeUpdate(self):
        """
        Update parameters after the conclusion of an episode.
        :return:
        """
        pass

    def setSaveInfo(self, special_name_tag=None):
        self.save_info = SaveInfo(self, special_name_tag=special_name_tag)

    def save(self):
        self.save_info.save()
