
import numpy as np

import agent
from agent.FFNN import FFNN


class Episode:
    """
    Hold all relevant values for a single episode.
    """

    def __init__(self, agent):
        self.rewards = None
        self.means_eligibility_traces_running_sum, self.stds_eligibility_traces_running_sum = None, None
        #  These matrices hold the running totals for (eligibility traces * reward decay) for each weight
        self.timeStep, self.episode_tag = None, None

        self.observation = None

        self.agent = agent

    def resetEpisode(self, episode_tag=None):  # TODO: Add tag
        """
        Reset the episode
        :param episode_tag: Episode identifier
        :return:
        """

        self.rewards = []

        self.timeStep = 0
        self.episode_tag = episode_tag  # parent class attributes
        self.observation = None

        self.means_eligibility_traces_running_sum = [np.zeros(m.shape) for m in self.agent.params.means]
        self.stds_eligibility_traces_running_sum = [np.zeros(s.shape) for s in self.agent.params.stds]

    def setObservation(self, observation):
        """
        Set the current observation
        :param observation:
        :return:
        """
        self.observation = observation.flatten()


class Params:
    """
    Hold parameters for the agent here.
    """

    def __init__(self, layers, nnet_bias=True):
        """
        :param layers: List of layer sizes (not including bias)
        """
        self.layers = layers
        self.nnet_bias = nnet_bias

        self.means = []
        self.stds = []

    def checkParams(self):
        """
        Checks to see if each layer has the correct number of parameters.
        :raise RuntimeError: Iff an error in the number of parameters was found
        :return:
        """
        error = False

        if self.means[0].shape != self.stds[0].shape or self.means[0].shape[1] != self.layers[0]:
            error = True

        for i in range(1, len(self.layers)):
            if self.means[i-1].shape != self.stds[i-1].shape:
                error = True

            if i != len(self.layers) - 1:
                if self.means[i].shape[1] != self.means[i - 1].shape[0] + int(self.nnet_bias):
                    error = True

                if self.means[i].shape[1] != self.layers[i - 1] + int(self.nnet_bias):
                    error = True

            else:
                if self.layers[i] != self.means[i-1].shape[0]:
                    error = True

        if error:
            raise RuntimeError("Param object with incorrect number of parameters.")

    def setMeans(self, means):
        self.means = means

    def setStds(self, stds):
        self.stds = stds

    def initializeRandom(self, random_bounds = ((-1, 1) , (0.05, 1))):
        """
        Set up parameters uniformly randomly.
        Note: If bias nodes are used, this function will add parameters accordingly.
        :param random_bounds: Bounds of means and variances for random setup
        :return:
        """

        bias = int(self.nnet_bias)

        for i, size in enumerate(self.layers):
            if i == 0:
                continue

            self.means.append(np.random.rand(size, self.layers[i-1] + bias)*(random_bounds[0][1] - random_bounds[0][0])
                              + random_bounds[0][0])

            self.stds.append(np.random.rand(size, self.layers[i-1] + bias)*(random_bounds[1][1] - random_bounds[1][0])
                              + random_bounds[1][0])


class History:
    """
    Holds information spanning multiple episodes, e.g. average past reward.
    """

    def __init__(self):
        self.past_rewards = []

    def getPastAverage(self, min_window=20, proportion=0.2):
        """
        Return the average cumulative reward per episode in a set number of past games.
        :param window: Number of past games to consider
        :return:
        """

        window = max(min_window, int(len(self.past_rewards)*proportion))

        return np.mean(self.past_rewards[-window:])


class SSRL(agent.Agent):
    """
    Implements Stochastic Synapse Reinforcement Learning
    """

    def __init__(self, layers, nnet_bias=True, nonlinearity=np.tanh, params=None, learning_rate=0.5, decay=None,
                 as_in_paper=True):

        """
        Provide information about the neural network architecture and set up basic data structures
        :param layers: Layer sizes (first element is flattened input size, last element is flattened output size)
        :param nnet_bias: True if bias should be added to non-output layers
        :param nonlinearity: Nonlinearity function for use in the network
        :param params: If not None, use these parameters
        :param learning_rate: Learning rate for mean and standard deviations in formula
        :param decay: Rate of reward decay for computing updates (None => No decay)
        """

        super().__init__()
        self.layers = layers
        self.nnet_bias = nnet_bias
        self.nonlinearity = nonlinearity
        self.learning_rate = learning_rate
        self.decay = decay

        self.NNET = FFNN(self.nonlinearity, self.nnet_bias)

        self.episode = Episode(self)
        self.params = None
        self.history = History()

        self.name = 'SSRL' if as_in_paper else 'SSRL_Jakob_variant'

        if params is not None:  # check if dimensions are correct

            params.checkParams()
            self.setParams(params)

    def setParams(self, params = None):
        """
        Set up parameter attributes (generate randomly unless otherwise specified)
        :param p: Params object or None
        :return:
        """

        if params is not None:
            self.params = params
        else:
            self.params = Params(self.layers, self.nnet_bias)
            self.params.initializeRandom()

    # Abstract methods

    def resetEpisode(self, episode_tag=None):  # TODO: Add tag

        self.episode.resetEpisode(episode_tag)

    def incrementTimeStep(self):

        self.episode.timeStep += 1

    def giveObservation(self, observation):

        self.episode.setObservation(observation)

    def act(self):

        """
        Sample parameters from normal distributions according to formulae, record inputs/activation, store update information
        and return NNET forward pass.
        :return: Action to take
        """

        action_weights = []

        for ind in range(len(self.layers)):
            if ind == len(self.layers) - 1:
                # weight matrices in NNET + 1 = number of layers
                continue

            mean = self.params.means[ind].flatten()
            cov = np.diag(self.params.stds[ind].flatten())

            layer_weights = np.random.multivariate_normal(mean, cov).reshape(self.params.means[ind].shape)
            #  sample the multivariate distribution and reshape it for the neural network

            action_weights.append(layer_weights)

        layer_activations = self.NNET.setWeights(action_weights)

        for weight_layer, weights in enumerate(action_weights):

            if self.decay is not None:
                self.episode.means_eligibility_traces_running_sum *= self.decay  # Apply reward decay
                self.episode.stds_eligibility_traces_running_sum *= self.decay

            diff_mean = (weights - self.params.means[weight_layer])

            self.episode.means_eligibility_traces_running_sum += layer_activations[weight_layer] * diff_mean

            self.episode.stds_eligibility_traces_running_sum += layer_activations[weight_layer] * \
            (np.abs(diff_mean) - self.params.stds[weight_layer])

        return np.argmax(layer_activations[-1])

    def giveReward(self, reward):

        self.episode.rewards.append(reward)

    def endOfEpisodeUpdate(self):

        """
        Apply the parameter update rules to means and standard deviations
        :return:
        """

        avg = self.history.getPastAverage()
        r = np.sum(self.episode.rewards)

        self.history.past_rewards.append(r)

        factor = self.learning_rate*(r - avg)

        #  Apply the parameter update rules
        for ind, m in enumerate(self.params.means):

            m += factor*self.episode.means_eligibility_traces_running_sum

        for ind, s in enumerate(self.params.stds):

            s += factor*self.episode.stds_eligibility_traces_running_sum

            self.params.stds[ind] = np.clip(s, 0.05, 1)











