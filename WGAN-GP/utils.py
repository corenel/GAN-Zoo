"""Utils for WGAN-GP."""

import random

import torch
from torch.autograd import Variable, grad

import params


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


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


def calc_gradient_penalty(D, real_data, fake_data):
    """Calculatge gradient penalty for WGAN-GP."""
    alpha = torch.rand(params.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = make_cuda(alpha)

    interpolates = make_variable(alpha * real_data + ((1 - alpha) * fake_data))

    disc_interpolates = D(interpolates)

    gradients = grad(outputs=disc_interpolates,
                     inputs=interpolates,
                     grad_outputs=make_cuda(
                         torch.ones(disc_interpolates.size())),
                     create_graph=True,
                     retain_graph=True,
                     only_inputs=True)[0]

    gradient_penalty = params.penalty_lambda * \
        ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
