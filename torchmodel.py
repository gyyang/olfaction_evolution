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
    return lambda x: x


class Layer(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Same as nn.Linear, except that weight matrix is constrained to be non-negative
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 sign_constraint=False,
                 weight_initializer=None,
                 weight_initial_value=None,
                 bias_initial_value=0,
                 pre_norm=None,
                 post_norm=None,
                 dropout=False,
                 dropout_rate=None,
                 prune_weak_weights=False,
                 prune_threshold=None,
                 feedforward_inh=False,
                 feedforward_inh_coeff=0,
                 recurrent_inh=False,
                 recurrent_inh_coeff=0,
                 recurrent_inh_step=1,
                 weight_norm=False,
                 ):
        super(Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight_initializer = weight_initializer
        if weight_initial_value:
            self.weight_init_range = weight_initial_value
        else:
            self.weight_init_range = 2. / in_features
        self.bias_initial_value = bias_initial_value
        self.sign_constraint = sign_constraint
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features))

        self.pre_norm = _get_normalization(pre_norm, num_features=out_features)
        self.activation = nn.ReLU()
        self.post_norm = _get_normalization(post_norm, num_features=out_features)

        if dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = lambda x: x

        self.prune_weak_weights = prune_weak_weights
        self.prune_threshold = prune_threshold

        self.reset_parameters()

        # self.w_dropout = nn.Dropout(p=0.1)
        self.w_dropout = nn.Identity()

        self.feedforward_inh = feedforward_inh
        self.feedforward_inh_coeff = feedforward_inh_coeff
        self.recurrent_inh = recurrent_inh
        self.recurrent_inh_coeff = recurrent_inh_coeff
        self.recurrent_inh_step = recurrent_inh_step

        # Weight normalization
        self.weight_norm = weight_norm

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
        if self.weight_initializer == 'constant':
            init.constant_(self.weight, self.weight_init_range)
        elif self.weight_initializer == 'uniform':
            low = self.prune_threshold if self.prune_weak_weights else 0.
            init.uniform_(self.weight, low, self.weight_init_range)
        elif self.weight_initializer == 'normal':
            init.normal_(self.weight, 0, self.weight_init_range)
        else:
            raise NotImplementedError('Unknown initializer',
                                      str(self.weight_initializer))

        if self.bias is not None:
            init.constant_(self.bias, self.bias_initial_value)

    @property
    def effective_weight(self):
        if self.sign_constraint:
            weight = torch.abs(self.weight)
        else:
            weight = self.weight

        if self.prune_weak_weights:
            not_pruned = (weight > self.prune_threshold)
            weight = weight * not_pruned

        if self.weight_norm:
            # Renormalize weights
            sums = torch.sum(weight, dim=1, keepdim=True)
            weight = torch.div(weight, sums)

        return weight

    def forward(self, input):
        # Random perturbation of weights
        # pre_act = F.linear(input, self.effective_weight, self.bias)
        # weight = self.w_dropout(self.effective_weight)
        if self.feedforward_inh:
            weight = (self.effective_weight - self.feedforward_inh_coeff *
                      torch.mean(self.effective_weight))
        else:
            weight = self.effective_weight
        pre_act = F.linear(input, weight, self.bias)
        pre_act_normalized = self.pre_norm(pre_act)

        output = self.activation(pre_act_normalized)

        if self.recurrent_inh:
            for i in range(self.recurrent_inh_step):
                # Single unit recurrent inhibition
                # No nonlinearity because it assumes positive input
                rec_inh = torch.mean(output, dim=1, keepdim=True)
                rec_inh = rec_inh * self.recurrent_inh_coeff
                output = self.activation(pre_act_normalized - rec_inh)

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
        self.multihead = config.label_type == 'multi_head_sparse'

        n_orn = config.N_ORN * config.N_ORN_DUPLICATION

        if config.receptor_layer:
            self.layer0 = Layer(
                config.N_ORN, n_orn,
                weight_initializer=config.initializer_or2orn,
                sign_constraint=config.sign_constraint_or2orn,
                bias=config.or_bias,
                weight_norm=config.or2orn_normalization,
            )

        self.layer1 = Layer(n_orn, config.N_PN,
                            weight_initializer=config.initializer_orn2pn,
                            weight_initial_value=config.initial_orn2pn,
                            sign_constraint=config.sign_constraint_orn2pn,
                            pre_norm=config.pn_norm_pre,
                            post_norm=config.pn_norm_post,
                            dropout=config.pn_dropout,
                            dropout_rate=config.pn_dropout_rate,
                            prune_weak_weights=config.pn_prune_weak_weights,
                            prune_threshold=config.pn_prune_threshold,
                            weight_norm=config.orn2pn_normalization,
                            )

        if config.skip_orn2pn or config.direct_glo:  # make these two the same
            init.eye_(self.layer1.weight.data)
            self.layer1.weight.requires_grad=False

        self.layer2 = Layer(config.N_PN, config.N_KC,
                            weight_initializer=config.initializer_pn2kc,
                            weight_initial_value=config.initial_pn2kc,
                            bias_initial_value=config.kc_bias,
                            sign_constraint=config.sign_constraint_pn2kc,
                            pre_norm=config.kc_norm_pre,
                            post_norm=config.kc_norm_post,
                            dropout=config.kc_dropout,
                            dropout_rate=config.kc_dropout_rate,
                            prune_weak_weights=config.kc_prune_weak_weights,
                            prune_threshold=config.kc_prune_threshold,
                            feedforward_inh=config.kc_ffinh,
                            feedforward_inh_coeff=config.kc_ffinh_coeff,
                            recurrent_inh=config.kc_recinh,
                            recurrent_inh_coeff=config.kc_recinh_coeff,
                            recurrent_inh_step=config.kc_recinh_step,
                            )

        if not config.train_kc_bias:
            self.layer2.bias.requires_grad = False

        self.layer3 = nn.Linear(config.N_KC, config.N_CLASS)
        self.loss = nn.CrossEntropyLoss()

        if self.multihead:
            self.layer3_2 = nn.Linear(config.N_KC, config.n_class_valence)
            self.loss_2 = nn.CrossEntropyLoss()

        self._readout = False

    def readout(self, is_readout=True):
        self._readout = is_readout

    def forward(self, x, target):
        # Process ORNs
        if self.config.receptor_layer:
            act0 = self.layer0(x)
        else:
            act0 = x
            if self.config.N_ORN_DUPLICATION > 1:
                act0 = act0.repeat(1, self.config.N_ORN_DUPLICATION)

        if self.config.ORN_NOISE_STD > 0:
            act0 += torch.randn_like(act0) * self.config.ORN_NOISE_STD

        act1 = self.layer1(act0)
        act2 = self.layer2(act1)
        y = self.layer3(act2)

        if self.multihead:
            target1, target2 = target[:, 0], target[:, 1]
            loss = self.loss(y, target1)

            y_2 = self.layer3_2(act2)
            loss_2 = self.loss_2(y_2, target2)
            with torch.no_grad():
                _, pred = torch.max(y, 1)
                acc = (pred == target1).sum().item() / target1.size(0)

                _, pred_2 = torch.max(y_2, 1)
                acc2 = (pred_2 == target2).sum().item() / target2.size(0)
            results = {'loss': loss + loss_2, 'acc': (acc + acc2) / 2,
                       'loss_1': loss, 'acc1': acc,
                       'loss_2': loss_2, 'acc2': acc2, 'kc': act2}
        else:
            # Regular network
            loss = self.loss(y, target)
            with torch.no_grad():
                _, pred = torch.max(y, 1)
                acc = (pred == target).sum().item() / target.size(0)
            results = {'loss': loss, 'acc': acc, 'kc': act2}

        if self._readout:
            results['glo'] = act1

        return results

    @property
    def w_or(self):
        if self.config.receptor_layer:
            return self.layer0.effective_weight.data.cpu().numpy().T
        else:
            return None

    @property
    def w_orn(self):
        # Transpose to be consistent with tensorflow default
        return self.layer1.effective_weight.data.cpu().numpy().T

    @property
    def w_glo(self):
        return self.layer2.effective_weight.data.cpu().numpy().T

    def save(self, epoch=None):
        save_path = self.config.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, 'model.pt')
        torch.save(self.state_dict(), fname)

    def load(self, epoch=None):
        save_path = self.config.save_path
        if not os.path.exists(save_path):
            # datasets are usually stored like path/files/experiment/model
            paths = ['.'] + os.path.normpath(save_path).split(os.path.sep)[-3:]
            save_path = os.path.join(*paths)

        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))
        fname = os.path.join(save_path, 'model.pt')
        if not torch.cuda.is_available():
            self.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(fname))

    def save_pickle(self, epoch=None):
        """Save model using pickle.

        This is quite space-inefficient. But it's easier to read out.
        """
        save_path = self.config.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, 'model.pkl')

        var_dict = dict()
        for name, param in self.named_parameters():
            var_dict[name] = param.data.cpu().numpy()

        if self.w_or is not None:
            var_dict['w_or'] = self.w_or
        var_dict['w_orn'] = self.w_orn
        var_dict['w_glo'] = self.w_glo
        with open(fname, 'wb') as f:
            pickle.dump(var_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model weights saved in path: %s" % save_path)