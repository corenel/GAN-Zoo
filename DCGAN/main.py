#!/usr/bin/env python3
"""Main code for DCGAN."""

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision

from models import Discriminator, Generator
from params import *
from utils import init_random_seed, init_weights

if __name__ == '__main__':
    # init random seed
    init_random_seed()

    # init models
    D = Discriminator(num_channels=num_channels,
                      conv_dim=d_conv_dim,
                      num_workers=num_workers)
    G = Generator(num_channels=num_channels,
                  z_dim=z_dim,
                  conv_dim=g_conv_dim,
                  num_workers=num_workers)

    # init weights of models
    D.apply(init_weights)
    G.apply(init_weights)

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        D.cuda()
        G.cuda()
