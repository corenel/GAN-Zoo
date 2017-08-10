"""Utils for GAN."""

import torch
from torch.autograd import Variable


def make_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
