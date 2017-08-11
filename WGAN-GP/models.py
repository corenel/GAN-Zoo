"""Models for DCGAN."""

import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from utils import init_weights


class Discriminator(nn.Module):
    """Model for Discriminator."""

    def __init__(self, num_channels, conv_dim, num_gpu):
        """Init for Discriminator model."""
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
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
                self.num_gpu > 1:
            out = nn.parallel.data_parallel(
                self.layer, input, range(self.num_gpu))
        else:
            out = self.layer(input)
        return out.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    """Model for Generator."""

    def __init__(self, num_channels, z_dim, conv_dim, num_gpu):
        """Init for Generator model."""
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
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
            # output layer, output (num_channels) x 64 x 64
            nn.ConvTranspose2d(conv_dim, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        """Forward step for Generator model."""
        if isinstance(input.data, torch.cuda.FloatTensor) and \
                self.num_gpu > 1:
            out = nn.parallel.data_parallel(
                self.layer, input, range(self.num_gpu))
        else:
            out = self.layer(input)
        # flatten output
        return out


def get_models(num_channels, d_conv_dim, g_conv_dim, z_dim, num_gpu,
               d_model_restore=None, g_model_restore=None):
    """Get models with cuda and inited weights."""
    D = Discriminator(num_channels=num_channels,
                      conv_dim=d_conv_dim,
                      num_gpu=num_gpu)
    G = Generator(num_channels=num_channels,
                  z_dim=z_dim,
                  conv_dim=g_conv_dim,
                  num_gpu=num_gpu)

    # init weights of models
    D.apply(init_weights)
    G.apply(init_weights)

    # restore model weights
    if d_model_restore is not None and os.path.exists(d_model_restore):
        D.load_state_dict(torch.load(d_model_restore))
    if g_model_restore is not None and os.path.exists(g_model_restore):
        G.load_state_dict(torch.load(g_model_restore))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        D.cuda()
        G.cuda()

    return D, G
