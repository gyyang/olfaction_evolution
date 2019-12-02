"""Model file."""

import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import math

from configs import FullConfig, SingleLayerConfig
import scipy.stats as st


def get_sparse_mask(nx, ny, non, complex=False, nOR=50):
    """Generate a binary mask.

    The mask will be of size (nx, ny)
    For all the nx connections to each 1 of the ny units, only non connections are 1.

    If complex == True, KCs cannot receive the connections from the same OR from duplicated ORN inputs.
    Assumed to be 'repeat' style duplication.

    Args:
        nx: int
        ny: int
        non: int, must not be larger than nx

    Return:
        mask: numpy array (nx, ny)
    """
    mask = np.zeros((nx, ny))

    if not complex:
        mask[:non] = 1
        for i in range(ny):
            np.random.shuffle(mask[:, i])  # shuffling in-place
    else:
        OR_ixs = [np.arange(i, nx, nOR) for i in range(nOR)] # only works for repeat style duplication
        for i in range(ny):
            ix = [np.random.choice(bag) for bag in OR_ixs]
            ix = np.random.choice(ix, non, replace=False)
            mask[ix,i] = 1
    return mask.astype(np.float32)


def _get_normalization(norm_type, num_features=None):
    if norm_type is not None:
        if norm_type == 'batch_norm':
            return nn.BatchNorm1d(num_features)
    else:
        return lambda x: x


class Layer(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Same as nn.Linear, except that weight matrix is constrained to be non-negative
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True,
                 sign_constraint=False, pre_norm=None, post_norm=None,
                 dropout=False, dropout_rate=None):
        super(Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sign_constraint = sign_constraint
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.pre_norm = _get_normalization(pre_norm, num_features=out_features)
        self.activation = nn.ReLU()
        self.post_norm = _get_normalization(post_norm, num_features=out_features)

        if dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        if self.sign_constraint:
            self._reset_sign_constraint_parameters()
        else:
            self._reset_parameters()

    def _reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _reset_sign_constraint_parameters(self):
        # the default for Linear, kaiming_uniform wouldn't work here
        init.eye_(self.weight)
        self.weight.data = self.weight.data * 0.5  # scaled identity matrix
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.sign_constraint:
            # weight is non-negative
            pre_act = F.linear(input, torch.abs(self.weight), self.bias)
        else:
            pre_act = F.linear(input, self.weight, self.bias)
        pre_act_normalized = self.pre_norm(pre_act)
        output = self.activation(pre_act_normalized)
        output_normalized = self.post_norm(output)
        output_normalized = self.dropout(output_normalized)
        return output_normalized


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class FullModel(nn.Module):
    def __init__(self, config=None):
        super(FullModel, self).__init__()
        if config is None:
            config = FullConfig

        self.config = config

        self.layer1 = Layer(config.N_PN, config.N_PN,
                            sign_constraint=config.sign_constraint_orn2pn,
                            pre_norm=config.pn_norm_pre,
                            post_norm=config.pn_norm_post,
                            dropout=config.pn_dropout,
                            dropout_rate=config.pn_dropout_rate)

        self.layer2 = Layer(config.N_PN, config.N_KC,
                            sign_constraint=config.sign_constraint_pn2kc,
                            pre_norm=config.kc_norm_pre,
                            post_norm=config.kc_norm_post,
                            dropout=config.kc_dropout,
                            dropout_rate=config.kc_dropout_rate)

        self.layer3 = nn.Linear(config.N_KC, config.N_CLASS)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, target):
        act1 = self.layer1(x)
        act2 = self.layer2(act1)
        y = self.layer3(act2)
        loss = self.loss(y, target)
        with torch.no_grad():
            _, pred = torch.max(y, 1)
            acc = (pred == target).sum().item() / target.size(0)
        return loss, acc