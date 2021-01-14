import torch
import torchmeta.modules as mmods
from collections import OrderedDict
import os
from typing import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import tools
import configs


class _linear_block(mmods.MetaLinear):
    def __init__(self,
                 in_features,
                 out_features,
                 sign_constraint,
                 prune,
                 bias_init_value=0):
        super(_linear_block, self).__init__(in_features, out_features)
        self.sign_constraint = sign_constraint
        self.bias_initial_value = bias_init_value
        self.prune = prune
        self.weight_init_range = 4. / in_features
        self.prune_threshold = 0.001
        self.reset_params()

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        weight = params.get('weight', None)
        return F.linear(input, self.effective_weight(weight), bias)

    def effective_weight(self, weight):
        if self.sign_constraint:
            weight = torch.abs(weight)

        if self.prune:
            weight[weight < self.prune_threshold] = 0
        return weight

    def reset_params(self):
        if self.sign_constraint:
            self._reset_sign_constraint_parameters()
        else:
            self._reset_parameters()

    def _reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def _reset_sign_constraint_parameters(self):
        init.uniform_(self.weight, self.prune_threshold, self.weight_init_range)
        init.constant_(self.bias, self.bias_initial_value)


class Layer(mmods.MetaModule):
    def __init__(self,
                 name: Text,
                 in_features: int,
                 out_features: int,
                 pre_norm: bool = False,
                 post_norm: bool = False,
                 sign_constraint: bool = False,
                 prune: bool = False,
                 dropout: bool = False,
                 dropout_rate: float = 0,
                 ):
        super().__init__()

        modules = []
        lin_mod = _linear_block(in_features=in_features,
                                out_features=out_features,
                                sign_constraint=sign_constraint,
                                prune=prune)
        modules += [(name + '_linear', lin_mod)]

        if pre_norm:
            bn_mod = mmods.MetaBatchNorm1d(num_features=out_features,
                                           momentum=0,
                                           track_running_stats=False)
            modules += [(name + '_bn_pre', bn_mod)]

        modules += [(name + '_relu', nn.ReLU())]

        if post_norm:
            bn_mod = mmods.MetaBatchNorm1d(num_features=out_features,
                                           momentum=0,
                                           track_running_stats=False)
            modules += [(name + '_bn_post', bn_mod)]

        if dropout:
            dropout_mod = nn.Dropout(p=dropout_rate)
            modules += [(name + '_dropout', dropout_mod)]

        self.block = mmods.MetaSequential(OrderedDict(modules))

    def forward(self, x, params=None):
        return self.block(x, params=self.get_subdict(params, 'block'))


class Model(mmods.MetaModule):
    def __init__(self, config: configs.MetaConfig = None):
        super().__init__()
        self.config = config
        n_orn = config.N_ORN * config.N_ORN_DUPLICATION

        self.layer1 = Layer('orn_layer',
                            n_orn,
                            config.N_PN,
                            sign_constraint=config.sign_constraint_orn2pn,
                            pre_norm=config.pn_norm_pre,
                            post_norm=config.pn_norm_post,
                            prune=config.prune,
                            dropout=config.pn_dropout,
                            dropout_rate=config.pn_dropout_rate,
                            )

        if config.skip_orn2pn:
            init.eye_(self.layer1.block.orn_layer_linear.weight.data)
            self.layer1.block.orn_layer_linear.weight.requires_grad=False

        self.layer2 = Layer('pn_layer',
                            config.N_PN,
                            config.N_KC,
                            sign_constraint=config.sign_constraint_pn2kc,
                            pre_norm=config.kc_norm_pre,
                            post_norm=config.kc_norm_post,
                            prune = config.prune,
                            dropout=config.kc_dropout,
                            dropout_rate=config.kc_dropout_rate,
                            )
        self.layer3 = mmods.MetaLinear(config.N_KC, config.N_CLASS)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, params=None):
        act0 = x
        if self.config.N_ORN_DUPLICATION > 1:
            act0 = act0.repeat(1, self.config.N_ORN_DUPLICATION)

        act1 = self.layer1(act0, params=self.get_subdict(params, 'layer1'))
        act2 = self.layer2(act1, params=self.get_subdict(params, 'layer2'))
        y = self.layer3(act2, params=self.get_subdict(params, 'layer3'))
        return y

    def save(self, epoch=None):
        save_path = self.config.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, 'model.pt')
        torch.save(self.state_dict(), fname)

    def save_pickle(self, epoch=None):
        var_dict = dict()
        for name, param in self.named_parameters():
            var_dict[name] = param.data.cpu().numpy()

        var_dict['w_orn'] = self.w_orn
        var_dict['w_glo'] = self.w_glo
        var_dict['w_out'] = self.w_out

        save_path = self.config.save_path
        tools.save_pickle(save_path, var_dict, epoch=epoch)
        print("Model weights saved in path: %s" % save_path)

    @property
    def w_orn(self):
        orn_weight = self.layer1.block.orn_layer_linear.effective_weight(
            self.layer1.block.orn_layer_linear.weight)
        # Transpose to be consistent with tensorflow default
        return orn_weight.cpu().detach().numpy().T

    @property
    def w_glo(self):
        pn_weight = self.layer2.block.pn_layer_linear.effective_weight(
            self.layer2.block.pn_layer_linear.weight)
        return pn_weight.cpu().detach().numpy().T

    @property
    def w_out(self):
        return self.layer3.weight.cpu().detach().numpy().T
