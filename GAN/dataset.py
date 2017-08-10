"""Dataset setting and data loader for GAN."""

import torch
from torchvision import datasets, transforms

from params import batch_size, dataset_mean, dataset_std

# image pre-processing
pre_process = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=dataset_mean,
                                                       std=dataset_std)])

# dataset and data loader
mnist_dataset = datasets.MNIST(root="../data/",
                               train=True,
                               transform=pre_process,
                               download=True)

mnist_dataloader = torch.utils.data.DataLoader(dataset=mnist_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
