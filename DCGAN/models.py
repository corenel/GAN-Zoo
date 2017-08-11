"""Models for DCGAN."""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Model for Discriminator."""

    def __init__(self, num_channels, conv_dim, num_workers):
        """Init for Discriminator model."""
        super(Discriminator, self).__init__()
        self.num_workers = num_workers
        self.layer = nn.Sequential(
            # 1st conv layer
            # input num_channels x 64 x 64, output conv_dim x 32 x 32
            nn.Conv2d(num_channels, conv_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd conv layer, output (conv_dim*2) x 16 x 16
            nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd conv layer, output (conv_dim*4) x 8 x 8
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th conv layer, output (conv_dim*8) x 4 x 4
            nn.Conv2d(conv_dim * 4, conv_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(conv_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        """Forward step for Discriminator model."""
        if isinstance(input.data, torch.cuda.FloatTensor) and \
                self.num_workers > 1:
            out = nn.parallel.data_parallel(
                self.layer, input, range(self.num_workers))
        else:
            out = self.layer(input)
        return out


class Generator(nn.Module):
    """Model for Generator."""

    def __init__(self, num_channels, z_dim, conv_dim, num_workers):
        """Init for Generator model."""
        super(Generator, self).__init__()
        self.num_workers = num_workers
        self.layer = nn.Sequential(
            # 1st deconv layer, input Z, output (conv_dim*8) x 4 x 4
            nn.ConvTranspose2d(z_dim, conv_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(True),
            # 2nd deconv layer, output (conv_dim*4) x 8 x 8
            nn.ConvTranspose2d(conv_dim * 8, conv_dim * \
                               4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(True),
            # 3rd deconv layer, output (conv_dim*2) x 16 x 16
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * \
                               2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(True),
            # 4th deconv layer, output (conv_dim) x 32 x 32
            nn.ConvTranspose2d(conv_dim * 2, conv_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
            # 2nd deconv layer, output (num_channels) x 64 x 64
            nn.ConvTranspose2d(conv_dim, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        """Forward step for Generator model."""
        if isinstance(input.data, torch.cuda.FloatTensor) and \
                self.num_workers > 1:
            out = nn.parallel.data_parallel(
                self.layer, input, range(self.num_workers))
        else:
            out = self.layer(input)
        # flatten output
        return out.view(-1, 1).squeeze(1)
