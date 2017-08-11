"""Dataset setting and data loader for WGAN-GP."""

import torchvision.datasets as dset
from torch.utils import data
from torchvision import transforms

from params import batch_size, data_root, dataset_mean, dataset_std, image_size

# image pre-processing
pre_process = transforms.Compose([transforms.Scale(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=dataset_mean,
                                                       std=dataset_std)])

# dataset and data loader
dataset = dset.CIFAR10(root=data_root,
                       transform=pre_process,
                       download=True
                       )

data_loader = data.DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True)
