"""Utils for WGAN."""

import random

import torch
from torch.autograd import Variable

import params


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def denormalize(x):
    """Invert normalization, and then convert array into image."""
    out = x * params.dataset_std_value + params.dataset_mean_value
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed():
    """Init random seed."""
    seed = None
    if params.manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = params.manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
