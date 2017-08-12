"""Dataset setting and data loader for WGAN-GP."""

import torchvision.datasets as dset
from torch.utils import data
from torchvision import transforms

import params

# image pre-processing
pre_process = transforms.Compose([transforms.Scale(params.image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      mean=params.dataset_mean,
                                      std=params.dataset_std)])

# dataset and data loader
dataset = dset.CIFAR10(root=params.data_root,
                       transform=pre_process,
                       download=True
                       )

data_loader = data.DataLoader(dataset=dataset,
                              batch_size=params.batch_size,
                              shuffle=True)
