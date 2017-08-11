"""Models for DCGAN."""

import torch.nn as nn


class Discriminator(nn.Module):
    """Model for Discriminator."""

    def __init__(self, num_workers):
        """Init for Discriminator model."""
        super(Discriminator, self).__init__()
        self.num_workers = num_workers
        self.layer = nn.Sequential(
        )

    def forward(self, input):
        """Forward step for Discriminator model."""
        if isinstance(input.data, torch.cuda.FloatTensor) and
        self.num_workers > 1:
            out = nn.parallel.data_parallel(
                self.layer, input, range(self.num_workers))
        else:
            out = self.layer(input)
        return out


class Generator(nn.Module):
    """Model for Generator."""

    def __init__(self, num_workers):
        """Init for Generator model."""
        super(Generator, self).__init__()
        self.num_workers = num_workers
        self.layer = nn.Sequential()

    def forward(self, x):
        """Forward step for Generator model."""
        if isinstance(input.data, torch.cuda.FloatTensor) and
        self.num_workers > 1:
            out = nn.parallel.data_parallel(
                self.layer, input, range(self.num_workers))
        else:
            out = self.layer(input)
        # flatten output
        return out.view(-1, 1).squeeze(1)
