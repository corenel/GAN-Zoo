"""Models for GAN."""

import torch.nn as nn


class Discriminator(nn.Module):
    """Model for Discriminator."""

    def __init__(self, input_size, hidden_size, output_size):
        """Init for Discriminator model."""
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_size, output_size),
                                   nn.Sigmoid())

    def forward(self, x):
        """Forward step for Discriminator model."""
        out = self.layer(x)
        return out


class Generator(nn.Module):
    """Model for Generator."""

    def __init__(self, input_size, hidden_size, output_size):
        """Init for Generator model."""
        super(Generator, self).__init__()
        self.layer = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_size, output_size),
                                   nn.Sigmoid())

    def forward(self, x):
        """Forward step for Generator model."""
        out = self.layer(x)
        return out
