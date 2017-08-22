"""Utils for WGAN-GP."""

import os
import random

import torch
import torchvision
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
    interpolates.requires_grad = True

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


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))


def save_fake_image(G, fixed_noise, filename):
    """Save fake images by Generator."""
    if not os.path.exists(params.data_root):
        os.makedirs(params.data_root)
    fake_images = G(fixed_noise)
    torchvision.utils.save_image(denormalize(fake_images.data),
                                 os.path.join(params.data_root, filename))
