"""Model file."""

import os

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import math

from configs import FullConfig, SingleLayerConfig, RNNConfig
import tools


def get_multiglo_mask(nx, ny, nglo):
    """Generates multiglomerular connections.

    The mask will be of size (nx, ny), connections are 1/nglo, no connections
    are 0. nglo dictates the number of glomeruli each ORN will connect to.

    The sum of the mask values add up to 1.
    """
    assert nglo > 0 and nglo <= nx
    mask = np.zeros((nx, ny))
    val = 1. / nglo
    mask[:, :nglo] = val
    for i in range(nx):
        mask[i,:] = np.roll(mask[i, :], shift=i)
    return mask.astype(np.float32)


def get_restricted_sparse_mask(nx, ny, n_on, n_patterns):
    """Generates a sparse binary mask with a fixed set of patterns defined by
    n_patterns. The number of connections per output neuron (ny) is defined by
    n_on.
    """
    templates = np.zeros((nx, n_patterns))
    templates[:n_on] = 1
    for i in range(n_patterns):
        np.random.shuffle(templates[:, i])  # shuffling in-place

    ixs = np.random.randint(low=0, high=n_patterns, size=ny)
    mask = templates[:, ixs]
    return mask.astype(np.float32)


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


class OlsenNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.exponent = 1
        self.r_max = nn.Parameter(torch.Tensor(1, num_features))
        self.rho = nn.Parameter(torch.Tensor(1, num_features))
        self.m = nn.Parameter(torch.Tensor(1, num_features))
        self.num_features = num_features
        nn.init.constant_(self.r_max, num_features / 2.)
        nn.init.constant_(self.rho, 0)
        nn.init.constant_(self.m, 0.99)

    def forward(self, input):
        r_max = torch.clamp(self.r_max, self.num_features / 10.,
                            self.num_features)
        rho = torch.clamp(self.rho, 0., 3.)
        m = torch.clamp(self.m, 0.05, 2.)

        input_sum = torch.sum(input, dim=-1, keepdim=True) + 1e-6
        input_exponentiated = input ** self.exponent
        numerator = r_max * input_exponentiated
        denominator = (input_exponentiated + rho +
                       (m * input_sum) ** self.exponent)
        return torch.div(numerator, denominator)


class FixActivityNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.rmax = num_features / 2.

    def forward(self, input):
        # input (batch_size, neurons)
        input_sum = torch.sum(input, dim=-1, keepdim=True) + 1e-6
        return self.rmax * torch.div(input, input_sum)


class MeanCenterNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # input (batch_size, neurons)
        return input - torch.mean(input, dim=-1, keepdim=True)


def _get_normalization(norm_type, num_features=None):
    if norm_type is not None:
        if norm_type == 'batch_norm':
            return nn.BatchNorm1d(num_features)
        elif norm_type == 'layer_norm':
            return nn.LayerNorm(num_features)
        elif norm_type == 'mean_center':
            return MeanCenterNorm()
        elif norm_type == 'fixed_activity':
            return FixActivityNorm(num_features)
        elif norm_type == 'olsen':
            return OlsenNorm(num_features)
        else:
            raise ValueError('Unknown norm type', norm_type)
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
                 weight_dropout=False,
                 weight_dropout_rate=None,
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
            # Maintains the mean activity when considering init kc bias -1
            self.weight_init_range = 4. / in_features
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

        if weight_dropout:
            self.weight_dropout = True
            self.w_dropout = nn.Dropout(p=weight_dropout_rate)
        else:
            self.weight_dropout = False

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

        if self.weight_dropout:
            weight = self.w_dropout(weight)

        pre_act = F.linear(input, weight, self.bias)  # (batch_size, neurons)
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


class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()

        self._readout = False
        # Record original weights for lesioning
        self._original_weights = {}

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

    def readout(self, is_readout=True):
        self._readout = is_readout

    def lesion_units(self, name, units, verbose=False, arg='outbound'):
        """Lesion units given by units.

        Args:
            name: name of the layer to lesion
            units : can be None, an integer index, or a list of integer indices
            verbose: bool
            arg: 'outbound' or 'inbound', lesion outgoing or incoming units
        """
        # Convert to numpy array
        if units is None:
            return
        elif not hasattr(units, '__iter__'):
            units = np.array([units])
        else:
            units = np.array(units)

        layer = getattr(self, name)
        if name not in self._original_weights:
            # weight not recorded yet
            self._original_weights[name] = layer.weight.clone().detach()

        original_weight = self._original_weights[name]

        layer.weight.copy_(original_weight)

        if arg == 'outbound':
            layer.weight.data[:, units] = 0
        elif arg == 'inbound':
            layer.weight.data[units, :] = 0
        else:
            raise ValueError('did not recognize lesion argument: {}'.format(arg))

        if verbose:
            print('Lesioned units:')
            print(units)


class FullModel(CustomModule):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = FullConfig()

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

        if not config.train_orn2pn:
            if config.n_glo > 0:
                layer1_w = get_multiglo_mask(n_orn, config.N_PN, config.n_glo)
                with torch.no_grad():
                    self.layer1.weight = nn.Parameter(torch.from_numpy(
                        layer1_w.T).float())
                    self.layer1.weight.requires_grad = False
            else:
                self.layer1.weight.requires_grad = False


        if config.skip_orn2pn:
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

        if not config.train_pn2kc:
            if config.sparse_pn2kc:
                layer2_w = get_sparse_mask(config.N_PN, config.N_KC,
                                           config.kc_inputs)
                with torch.no_grad():
                    self.layer2.weight = nn.Parameter(torch.from_numpy(
                        layer2_w.T).float())
            elif config.restricted_sparse_pn2kc:
                layer2_w = get_restricted_sparse_mask(
                    config.N_PN,
                    config.N_KC,
                    config.kc_inputs,
                    config.n_restricted_patterns)
                with torch.no_grad():
                    self.layer2.weight = nn.Parameter(torch.from_numpy(
                        layer2_w.T).float())
                    self.layer2.weight.requires_grad = False
            else:
                self.layer2.weight.requires_grad = False



        self.layer3 = nn.Linear(config.N_KC, config.N_CLASS)
        self.loss = nn.CrossEntropyLoss()

        if self.multihead:
            self.layer3_2 = nn.Linear(config.N_KC, config.n_class_valence)
            self.loss_2 = nn.CrossEntropyLoss()

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

    @property
    def w_out(self):
        return self.layer3.weight.data.cpu().numpy().T

    @property
    def w_out2(self):
        if self.multihead:
            return self.layer3_2.weight.data.cpu().numpy().T
        else:
            return None

    def save_pickle(self, epoch=None):
        """Save model using pickle.

        This is quite space-inefficient. But it's easier to read out.
        """
        var_dict = dict()
        for name, param in self.named_parameters():
            var_dict[name] = param.data.cpu().numpy()

        if self.w_or is not None:
            var_dict['w_or'] = self.w_or
        var_dict['w_orn'] = self.w_orn
        var_dict['w_glo'] = self.w_glo
        var_dict['w_out'] = self.w_out
        if self.multihead:
            var_dict['w_out2'] = self.w_out2

        save_path = self.config.save_path
        tools.save_pickle(save_path, var_dict, epoch=epoch)
        print("Model weights saved in path: %s" % save_path)


class SingleLayerModel(CustomModule):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = SingleLayerConfig()

        self.config = config

        n_orn = config.N_ORN * config.N_ORN_DUPLICATION

        self.layer = nn.Linear(n_orn, config.N_CLASS)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, target):
        # Process ORNs

        act0 = x
        if self.config.N_ORN_DUPLICATION > 1:
            act0 = act0.repeat(1, self.config.N_ORN_DUPLICATION)
        y = self.layer(act0)

        # Regular network
        loss = self.loss(y, target)
        with torch.no_grad():
            _, pred = torch.max(y, 1)
            acc = (pred == target).sum().item() / target.size(0)
        results = {'loss': loss, 'acc': acc}

        if self._readout:
            results['y'] = y

        return results

    def save_pickle(self, epoch=None):
        """Save model using pickle.

        This is quite space-inefficient. But it's easier to read out.
        """
        var_dict = dict()
        for name, param in self.named_parameters():
            var_dict[name] = param.data.cpu().numpy()

        save_path = self.config.save_path
        tools.save_pickle(save_path, var_dict, epoch=epoch)
        print("Model weights saved in path: %s" % save_path)


class RNNModel(CustomModule):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = RNNConfig

        self.config = config
        self.n_steps = config.TIME_STEPS
        self.hidden_units = config.NEURONS
        hidden_units = self.hidden_units

        self.rnn = Layer(
            hidden_units, hidden_units,
            weight_initializer=config.initializer_rec,
            weight_initial_value=config.initial_rec,
            sign_constraint=config.sign_constraint_rec,
            pre_norm=config.rec_norm_pre,
            post_norm=config.rec_norm_post,
            dropout=config.rec_dropout,
            # dropout=False,
            dropout_rate=config.rec_dropout_rate,
            weight_dropout=config.weight_dropout,
            weight_dropout_rate=config.weight_dropout_rate,
            prune_weak_weights=config.prune_weak_weights,
            prune_threshold=config.prune_threshold,
        )
        if config.diagonal:
            self.rnn.weight.data.fill_diagonal_(1.)  # set diagonal to 1

        self.output = nn.Linear(hidden_units, config.N_CLASS)
        self.loss = nn.CrossEntropyLoss()

    @property
    def w_rnn(self):
        return self.rnn.effective_weight.data.cpu().numpy().T

    @property
    def w_out(self):
        return self.output.weight.data.cpu().numpy().T

    def forward(self, x, target):
        # Process ORNs
        act0 = x
        if self.config.N_ORN_DUPLICATION > 1:
            act0 = act0.repeat(1, self.config.N_ORN_DUPLICATION)

        if self.config.ORN_NOISE_STD > 0:
            act0 += torch.randn_like(act0) * self.config.ORN_NOISE_STD

        # Set first N_ORN neurons to be odor activated
        act1 = torch.zeros([act0.shape[0], self.hidden_units],
                           dtype=act0.dtype).to(act0.device)
        act1[:, :act0.shape[1]] = act0

        results = dict()
        if self._readout:
            results['rnn_outputs'] = [act1.cpu().numpy()]

        act_sum = 0.

        for i in range(self.n_steps):
            if not self.config.allow_reactivation:
                # Keep track of cumulative activation
                act_sum = act_sum + act1

            act1 = self.rnn(act1)

            if not self.config.allow_reactivation:
                # TODO: doesn't work, remove soon
                # prevent neurons activated before from being re-activated
                # act1 = act1 * (1 - torch.heaviside(act_sum, torch.tensor(0.)))
                act1 = act1 * (act_sum < 0.001)

            if self._readout:
                results['rnn_outputs'].append(act1.cpu().numpy())

        # TODO: temp
        # act1 = self.dropout(act1)

        y = self.output(act1)

        # Regular network
        loss = self.loss(y, target)
        with torch.no_grad():
            _, pred = torch.max(y, 1)
            acc = (pred == target).sum().item() / target.size(0)
        results.update({'loss': loss, 'acc': acc})

        return results

    def save_pickle(self, epoch=None):
        """Save model using pickle.

        This is quite space-inefficient. But it's easier to read out.
        """
        var_dict = dict()
        for name, param in self.named_parameters():
            var_dict[name] = param.data.cpu().numpy()

        var_dict['w_rnn'] = self.w_rnn
        var_dict['w_out'] = self.w_out

        save_path = self.config.save_path
        tools.save_pickle(save_path, var_dict, epoch=epoch)
        print("Model weights saved in path: %s" % save_path)


def get_model(config):
    if config.model == 'full':
        return FullModel(config=config)
    elif config.model == 'rnn':
        return RNNModel(config=config)
    elif config.model == 'singlelayer':
        return SingleLayerModel(config=config)
    else:
        raise ValueError('Unknown model type', config.model)
