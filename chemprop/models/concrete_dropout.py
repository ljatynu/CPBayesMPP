from copy import deepcopy

import torch
from torch import nn
import numpy as np

# Partially based on https://github.com/yaringal/ConcreteDropout

class ConcreteDropout(nn.Module):
    def __init__(self, layer, reg_acc, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1, depth=1, train_strategy='cd'):
        """
        We use transfer_layer to contain the weight we transfer from the pre-trained model, in order to calculate the downstrean KL divergence.
        """
        super(ConcreteDropout, self).__init__()


        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.layer = layer
        self.transfer_layer = None
        self.train_strategy = train_strategy

        self.reg_acc = reg_acc
        self.reg_acc.notify_loss(depth)

        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(self.init_min, self.init_max))

    def forward(self, x):
        p = torch.sigmoid(self.p_logit)

        out = self.layer(self._concrete_dropout(x, p))

        # Calculate KL divergence
        if self.training:
            sum_of_square = 0
            if self.train_strategy == 'CPBayesMPP' or self.train_strategy == 'CPBayesMPP+OOD' or self.train_strategy == 'CPBayesMPP+AL':
                for (param, transfer_param) in zip(self.layer.parameters(), self.transfer_layer.parameters()):
                    sum_of_square += torch.sum(torch.pow(param - transfer_param, 2))  # Eqn (16) in the paper

            else:
                for param in self.layer.parameters():
                    sum_of_square += torch.sum(torch.pow(param, 2))

            # Pretraining: weights_regularizer = 1.0 * 1 / train_data_size
            # Downstream: weights_regularizer = 5.0 * 1 / train_data_size
            weights_regularizer = self.weight_regularizer * sum_of_square * (1 - p)

            dropout_regularizer = p * torch.log(p)
            dropout_regularizer += (1. - p) * torch.log(1. - p)

            input_dimensionality = x[0].numel()  # Number of elements of first item in batch
            dropout_regularizer *= self.dropout_regularizer * input_dimensionality

            regularization = weights_regularizer + dropout_regularizer

            self.reg_acc.add_loss(regularization)

        return out

    def _concrete_dropout(self, x, p):

        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                     - torch.log(1 - p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x

    def record_transfer_layer(self, transfer_layer):
        self.transfer_layer = deepcopy(transfer_layer)

    def reset_dropout_rate(self):
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(self.init_min, self.init_max))


class RegularizationAccumulator:
    def __init__(self):
        self.i = 0
        self.size = 0

    def notify_loss(self, depth):
        self.size += depth

    def initialize(self, cuda):
        self.arr = torch.empty(self.size)
        if cuda:
            self.arr = self.arr.cuda()

    def add_loss(self, loss):
        self.arr[self.i] = loss
        self.i += 1

    def get_sum(self):
        sum = torch.sum(self.arr)

        # reset index and computational graph
        self.i = 0
        self.arr = self.arr.detach()

        return sum
