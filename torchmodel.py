"""Model file."""

import os
import pickle

import numpy as np
import torch
from configs import FullConfig, SingleLayerConfig
import scipy.stats as st

class FullModel(torch.nn.Module):
    def __init__(self, config=None):
        super(FullModel, self).__init__()
        if config is None:
            config = FullConfig

        self.config = config

        self.layer1 = torch.nn.Linear(50, 50)
        self.layer2 = torch.nn.Linear(50, 2500)
        self.layer3 = torch.nn.Linear(2500, 100)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, target):
        act1 = self.layer1(x)
        act2 = self.layer2(act1)
        y = self.layer3(act2)
        loss = self.loss(y, target)
        with torch.no_grad():
            _, pred = torch.max(y, 1)
            acc = (pred == target).sum().item() / target.size(0)
        return loss, acc