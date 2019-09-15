import torch
from torch import from_numpy
import torch.nn.functional as F
from agent.Agent import Agent
from agent.SSRL import SSRL, Episode, History, Params
from agent.torchFFNN import FFNN



'''
A note on how params are handled in the torch case: 

Params will be allocated as numpy arrays and converted to torch tensors. The underlying numpy object
will not be destroyed, but both the array and the tensor will share the same space in memory, so changes to either
object affect both equally. 


Note: Numpy params should only be used when initializing the agent! Use tensors elsewhere

'''

class DEEPSSRL(SSRL):
    """
    Implements multi-layered Stochastic Synapse Reinforcement Learning
    """
    def __init__(self, layers=(100, 100, 50, 4), nnet_bias=True, nonlinearity=torch.tanh, params=None, learning_rate=0.5,
                 decay=None):
        """
         Provide information about the neural network architecture and set up basic data structures
         :param layers: Layer sizes (first element is flattened input size, last element is flattened output size)
         :param nnet_bias: True if bias should be added to non-output layers
         :param nonlinearity: Nonlinearity function for use in the network
         :param params: If not None, use these parameters
         :param learning_rate: Learning rate for mean and standard deviations in formula
         :param decay: Rate of reward decay for computing updates (None => No decay)
         """

        super(SSRL).__init__()  # get Agent attributes

        self.layers = layers
        self.nnet_bias = nnet_bias
        self.nonlinearity = nonlinearity
        self.learning_rate = learning_rate
        self.decay = decay

        self.default_name = "DEEPSSRL"
        self.resetName()

        self.NNET = FFNN(layers, self.nonlinearity, self.nnet_bias)

        self.episode = Episode(self)
        self.params = params
        self.history = History()

        if params is not None:
            params.checkParams()
            self.setParams(params)
            self.params.copy_torch_params()

    def setParams(self, params=None):
        super().setParams(params)
        self.params.copy_torch_params()

    def giveObservation(self, observation):
        observation = observation.flatten()
        observation = from_numpy(observation).float()
        observation.unsqueeze(0)
        self.episode.setObservation(observation)

    def get_action_weights(self):
        a = super().get_action_weights()
        a = [from_numpy(i).float() for i in a]
        for w in a:
            w.requires_grad = True
        return a

    def act(self):

        observation = self.episode.getObservation()

        if observation is None:
            raise RuntimeError("Called act on None observation")

        action_weights = self.get_action_weights()
        self.NNET.setWeights(action_weights)

        layer_activations = self.NNET.forward(observation)
        action = int(torch.argmax(layer_activations[-1]) + 1)

        if self.decay is not None:
            self.episode.means_eligibility_traces_running_sum *= self.decay  # Apply reward decay
            self.episode.stds_eligibility_traces_running_sum *= self.decay

        for weight_layer, weights in enumerate(action_weights):

            diff_mean = weights - self.params.means_torch[weight_layer]
            chi = torch.abs(layer_activations[weight_layer])

            for out_val in range(self.layers[-1]):  # backpropagate through each of the output nodes
                self.NNET.zero_grad()
                layer_activations[-1][out_val].backward()

                self.episode.means_eligibility_traces_running_sum[weight_layer] += (
                        chi * diff_mean * weights.grad)
                self.episode.stds_eligibility_traces_running_sum[weight_layer] += (
                        chi * (torch.abs(diff_mean) - self.params.stds_torch[weight_layer]) * weights.grad)



        return action



