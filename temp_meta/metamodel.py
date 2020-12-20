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

class _linear_block(mmods.MetaLinear):
    def __init__(self,
                 in_features,
                 out_features,
                 sign_constraint,
                 bias_init_value=0):
        super(_linear_block, self).__init__(in_features, out_features)
        self.sign_constraint = sign_constraint
        self.bias_initial_value = bias_init_value
        self.weight_init_range = 4. / in_features
        self.reset_params()

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        weight = params.get('weight', None)
        if self.sign_constraint:
            effective_weight = torch.abs(weight)
        else:
            effective_weight = weight
        return F.linear(input, effective_weight, bias)

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
        init.uniform_(self.weight, 0, self.weight_init_range)
        init.constant_(self.bias, self.bias_initial_value)


class Layer(mmods.MetaModule):
    def __init__(self,
                 name: Text,
                 in_features: int,
                 out_features: int,
                 pre_norm: bool = False,
                 post_norm: bool = False,
                 sign_constraint: bool = False,
                 dropout: bool = False,
                 dropout_rate: float = 0,
                 ):
        super().__init__()

        modules = []
        lin_mod = _linear_block(in_features=in_features,
                                out_features=out_features,
                                sign_constraint=sign_constraint)
        modules += [(name + '_linear', lin_mod)]

        if pre_norm:
            bn_mod = mmods.MetaBatchNorm1d(num_features=out_features,
                                           momentum=1,
                                           track_running_stats=False)
            modules += [(name + '_bn_pre', bn_mod)]

        modules += [(name + '_relu', nn.ReLU())]

        if post_norm:
            bn_mod = mmods.MetaBatchNorm1d(num_features=out_features,
                                           momentum=1,
                                           track_running_stats=False)
            modules += [(name + '_bn_post', bn_mod)]

        if dropout:
            dropout_mod = nn.Dropout(p=dropout_rate)
            modules += [(name + '_dropout', dropout_mod)]

        self.block = mmods.MetaSequential(OrderedDict(modules))

    def forward(self, x, params=None):
        return self.block(x, params=self.get_subdict(params, 'block'))


class model(mmods.MetaModule):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        n_orn = config.N_ORN * config.N_ORN_DUPLICATION

        self.layer1 = Layer('orn_layer',
                            n_orn,
                            config.N_PN,
                            sign_constraint=config.sign_constraint_orn2pn,
                            pre_norm=config.pn_norm_pre,
                            post_norm=config.pn_norm_post,
                            dropout=config.pn_dropout,
                            dropout_rate=config.pn_dropout_rate,
                            )

        self.layer2 = Layer('pn_layer',
                            config.N_PN,
                            config.N_KC,
                            sign_constraint=config.sign_constraint_pn2kc,
                            pre_norm=config.kc_norm_pre,
                            post_norm=config.kc_norm_post,
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
        """Save model using pickle.

        This is quite space-inefficient. But it's easier to read out.
        """
        var_dict = dict()
        for name, param in self.named_parameters():
            var_dict[name] = param.data.cpu().numpy()

        var_dict['w_orn'] = torch.abs(
            self.layer1.block.orn_layer_linear.weight).detach().numpy()
        var_dict['w_glo'] = torch.abs(
            self.layer2.block.pn_layer_linear.weight).detach().numpy()
        var_dict['w_out'] = self.layer3.weight.detach().numpy()

        save_path = self.config.save_path
        tools.save_pickle(save_path, var_dict, epoch=epoch)
        print("Model weights saved in path: %s" % save_path)





