import torch
from torch import from_numpy
import torch.nn.functional as F
from agent.Agent import Agent
from agent.SSRL import SSRL, Episode, History, Params
from agent.torchFFNN import FFNN
import numpy as np
from run_files.config import dtype, cuda



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
                 decay=None, use_abs_chi=True, diff_by_all=True, start_at_1=True, use_special_binary_output=False,
                 use_abs_val_of_grad=False, use_argmax_out=True):
        """
         Provide information about the neural network architecture and set up basic data structures
         :param layers: Layer sizes (first element is flattened input size, last element is flattened output size)
         :param nnet_bias: True if bias should be added to non-output layers
         :param nonlinearity: Nonlinearity function for use in the network
         :param params: If not None, use these parameters
         :param learning_rate: Learning rate for mean and standard deviations in formula
         :param decay: Rate of reward decay for computing updates (None => No decay)
         :param use_abs_chi: If True, uses |x_i(k)| instead of |x_i(k)| in the formula
         :param diff_by_all: If True, sum eligibility traces using the gradients for all output nodes, not just the argmax
         :param use_special_binary_output: see SSRL doc
         :param use_abs_val_of_grad: If true, apply the absolute value of the gradient derivatives in update formula instead
         of gradients.
         :param start_at_1: See SSRL
         :param use_argmax_out: If True, use the arg-max as the output (in consideration of start_at_1). Else give the whole vector as a np.array
         """

        super(SSRL).__init__()  # get Agent attributes

        self.layers = layers
        self.nnet_bias = nnet_bias
        self.nonlinearity = nonlinearity
        self.learning_rate = learning_rate
        self.decay = decay
        self.use_abs_update = use_abs_chi
        self.diff_by_all = diff_by_all

        self.default_name = "DEEPSSRL"
        self.resetName()
        self.start_at_1 = start_at_1
        self.use_special_binary_output = use_special_binary_output
        self.use_abs_val_of_grad = use_abs_val_of_grad
        self.use_argmax_out = use_argmax_out

        self.NNET = FFNN(layers, self.nonlinearity, self.nnet_bias, apply_final_activation=not self.use_argmax_out)   # .cuda()
        if cuda:
            self.NNET.cuda()

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
        observation = from_numpy(observation).type(dtype)
        observation.unsqueeze(0)
        self.episode.setObservation(observation)

    def get_action_weights(self):
        a = super().get_action_weights()
        a = [from_numpy(i).type(dtype) for i in a]
        for w in a:
            w.requires_grad = True
        return a

    def apply_act_updates(self, action_weights, layer_activations, action):
        """
        Apply updates from a single action step
        :param action_weights: Sampled weights used in this step
        :param layer_activations: All layer activations in the ANN
        :param action: The output of the agent at this step
        :return:
        """
        if self.decay is not None:
            self.episode.means_eligibility_traces_running_sum *= self.decay  # Apply reward decay
            self.episode.stds_eligibility_traces_running_sum *= self.decay

        for out_val in range(self.layers[-1]):  # backpropagate through each of the output nodes
            if not self.diff_by_all and out_val != (action - int(self.start_at_1)):
                assert self.use_argmax_out is True
                continue

            self.NNET.zero_grad()
            layer_activations[-1][out_val].backward(retain_graph=False if out_val == self.layers[-1] - 1 else True)
            # retain the computational graph for intermediate for all but the last output

            for weight_layer, param_object in enumerate(action_weights):

                bias = (weight_layer % 2 == 1)  # this weight layer holds bias weights
                weight_layer = weight_layer//2

                diff_mean = param_object - self.params.means_torch[weight_layer][:, :-1] if not bias else (
                    param_object - self.params.means_torch[weight_layer][:, -1]
                )
                diff_std = (torch.abs(diff_mean) - self.params.stds_torch[weight_layer][:, :-1]) if not bias else (
                    (torch.abs(diff_mean) - self.params.stds_torch[weight_layer][:, -1])
                )

                if bias:
                    chi = torch.ones(param_object.size())
                else:
                    if self.use_abs_update:
                        chi = torch.abs(layer_activations[weight_layer])
                    else:
                        chi = layer_activations[weight_layer]

                """
                We store all means and stds in a big matrix, and the last column are the bias values (we used to append 1. 
                to the input vector to absorb the bias. Now, we must manually separate the weight block from the bias column
                and perform the updates separately.             
                """

                update_means = self.episode.means_eligibility_traces_running_sum[weight_layer][:, :-1] if not bias else (
                    self.episode.means_eligibility_traces_running_sum[weight_layer][:, -1]
                )
                update_stds = self.episode.stds_eligibility_traces_running_sum[weight_layer][:, :-1] if not bias else (
                    self.episode.stds_eligibility_traces_running_sum[weight_layer][:, -1]
                )

                grad = torch.abs(param_object.grad) if self.use_abs_val_of_grad else param_object.grad

                update_means += chi * (diff_mean * grad)
                update_stds += chi * (diff_std * grad)


    def act(self, test=False):

        observation = self.episode.getObservation()

        if observation is None:
            raise RuntimeError("Called act on None observation")

        action_weights = self.get_action_weights()
        self.NNET.setWeights(action_weights)

        layer_activations = self.NNET.forward(observation)
        if self.use_argmax_out:
            action = int(torch.argmax(layer_activations[-1]) + int(self.start_at_1))  # FIXME Needs to be +1 for Sokoban

        else:
            action = layer_activations[-1].clone().detach().numpy()

        if self.use_special_binary_output:
            assert self.use_argmax_out is False
            assert layer_activations[-1].numel() == 1
            action = int(layer_activations[-1][0] >= 0)

        if test:
            # don't change anything
            return action

        self.apply_act_updates(action_weights, layer_activations, action)

        return action

    def endOfEpisodeUpdate(self, step_distance, test=False):

        """
        Apply the parameter update rules to means and standard deviations
        :return:
        """

        r, avg = self.reward_get(step_distance, test=test)

        if test:
            # don't change or update anything
            return

        factor = self.learning_rate*(r - avg)

        #  Apply the parameter update rules
        for ind, m in enumerate(self.params.means_torch):

            self.params.means_torch[ind] = m + self.learning_rate*(r - avg)*self.episode.means_eligibility_traces_running_sum[ind]
            self.params.means[ind] = self.params.means_torch[ind].detach().numpy()

        for ind, s in enumerate(self.params.stds_torch):

            self.params.stds_torch[ind] = s + self.learning_rate*(r - avg)*self.episode.stds_eligibility_traces_running_sum[ind]
            self.params.stds_torch[ind] = torch.clamp(self.params.stds_torch[ind], min=0.05, max=1.)
            self.params.stds[ind] = self.params.stds_torch[ind].detach().numpy()



