#!/usr/bin/env python3
"""Main code for DCGAN."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from models import Discriminator, Generator
from params import *

if __name__ == '__main__':
    # init models
    D = Discriminator(num_channels=num_channels,
                      conv_dim=d_conv_dim,
                      num_workers=num_workers)
    G = Generator(num_channels=num_channels,
                  z_dim=z_dim,
                  conv_dim=g_conv_dim,
                  num_workers=num_workers)

    # check if cuda is available
    if torch.cuda.is_available():
        D.cuda()
        G.cuda()
