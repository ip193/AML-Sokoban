import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FFNN(nn.Module):
    """
    torch-based NNET for backpropagation
    """

    def __init__(self, layer_sizes, nonlinearity, bias=True):
        """
        Customize pytorch-style ANN
        :param layer_sizes: Layer sizes (should begin with 100 and end with 4)
        :param nonlinearity: Of pytorch type
        :param bias: Add bias node to layers?
        """
        super(FFNN, self).__init__()
        self.layers = []  # holds the actual computational layers
        self.nonlinearity = nonlinearity
        self.bias = bias

        for ind in range(len(layer_sizes))[:-1]:
            #  add the fully-connected layers
            setattr(self, 'fc' + str(ind+1), nn.Linear(layer_sizes[ind], layer_sizes[ind + 1], bias=bias))
            self.layers.append(getattr(self, 'fc' + str(ind+1)))

    def setWeights(self, weights):
        """
        Set each layer's .weight.data attributes (using the result of Gaussian sampling) and replace them with
        the parameter object from the nn.Module
        :param params:
        :return:
        """
        new_weights = []
        for ind, p in enumerate(self.parameters()):
            bias = ((ind + 1) % 2 == 0)
            ind = ind//2  # we need to separate the bias column from the weights
            if not bias:
                p.data = weights[ind][:, :-1]
            else:
                p.data = weights[ind][:, -1].view(-1)
            new_weights.append(p)

        weights.clear()
        weights += new_weights  # here we replace the tensors produced by Gaussian sampling with the objects
        # stored in the NNET's parameters because we want to access their .grad values later

    def forward(self, input):
        """
        Note: File preparation (.unsqueeze etc.) should be handled by
        Episode.giveObservation()
        :param input:
        :return:
        """

        all_activations = []
        activation = input
        activation.requires_grad = True

        for ind, l in enumerate(self.layers):

            # Note: In contrast to the standard FFNN, we do not add the bias value in-place,
            # because torch methods implicitly add the bias node.

            all_activations.append(activation)
            activation = l(activation)  # bias has already been configured for this layer
            if ind != len(self.layers)-1:
                activation = self.nonlinearity(activation)
                # we want the argmax and tanh is strictly increasing

        all_activations.append(activation)
        return all_activations




