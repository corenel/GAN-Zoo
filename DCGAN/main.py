#!/usr/bin/env python3
"""Main code for DCGAN."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from models import get_models
from params import *
from utils import init_random_seed

if __name__ == '__main__':
    # init random seed
    init_random_seed()

    # init models
    D, G = get_models(num_channels, d_conv_dim, g_conv_dim, z_dim, num_workers)
