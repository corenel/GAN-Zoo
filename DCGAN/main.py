#!/usr/bin/env python3
"""Main code for DCGAN."""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from data_loader import data_loader
from models import get_models
from params import *
from utils import denormalize, init_random_seed, make_variable

if __name__ == '__main__':
    # init random seed
    init_random_seed()

    # init models
    D, G = get_models(num_channels, d_conv_dim, g_conv_dim, z_dim, num_gpu,
                      d_model_restore, g_model_restore)

    # setup optimizer
    criterion = nn.BCELoss()
    if torch.cuda.is_available():
        criterion.cuda()
    d_optimizer = optim.Adam(
        D.parameters(), lr=learning_rate, betas=(beta1, beta2))
    g_optimizer = optim.Adam(
        G.parameters(), lr=learning_rate, betas=(beta1, beta2))

    # training
    fixed_noise = make_variable(torch.randn(
        batch_size, z_dim, 1, 1).normal_(0, 1))

    for epoch in range(num_epochs):
        for step, (images, _) in enumerate(data_loader):
            batch_size = images.size(0)
            real_labels = make_variable(torch.ones(batch_size))
            fake_labels = make_variable(torch.zeros(batch_size))

            # training discriminator
            D.zero_grad()

            images = make_variable(images)
            noise = make_variable(torch.randn(
                batch_size, z_dim, 1, 1).normal_(0, 1))

            d_pred_real = D(images)
            d_loss_real = criterion(d_pred_real, real_labels)

            fake_images = G(noise)
            d_pred_fake = D(fake_images)
            d_loss_fake = criterion(d_pred_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # training generator
            D.zero_grad()
            G.zero_grad()

            noise = make_variable(torch.randn(
                batch_size, z_dim, 1, 1).normal_(0, 1))

            fake_images = G(noise)
            d_pred_fake = D(fake_images)
            g_loss = criterion(d_pred_fake, real_labels)
            g_loss.backward()

            g_optimizer.step()

            # print info
            if ((step + 1) % log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={} g_loss={} D(x)={} D(G(z))={}"
                      .format(epoch + 1,
                              num_epochs,
                              step + 1,
                              len(data_loader),
                              d_loss.data[0],
                              g_loss.data[0],
                              d_loss_real.data[0],
                              d_loss_fake.data[0]))

            # save fake images
            if ((step + 1) % sample_step == 0):
                fake_images = G(fixed_noise)
                torchvision.utils.save_image(denormalize(fake_images.data),
                                             "../data/DCGAN-fake-{}-{}.png"
                                             .format(epoch + 1, step + 1))

        # save the model parameters
        if ((epoch + 1) % save_step == 0):
            torch.save(D.state_dict(), os.path.join(
                model_root, "DCGAN-discriminator-{}.pkl".format(epoch + 1)))
            torch.save(G.state_dict(), os.path.join(
                model_root, "DCGAN-generator-{}.pkl".format(epoch + 1)))
