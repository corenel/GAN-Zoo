"""Utils for GAN."""

import torch
from torch.autograd import Variable

from params import dataset_mean_value, dataset_std_value


def make_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def denormalize(x):
    out = x * dataset_std_value + dataset_mean_value
    return out.clamp(0, 1)
