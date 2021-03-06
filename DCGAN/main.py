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
    ####################
    # 1. setup network #
    ####################

    # init random seed
    init_random_seed()

    # init models
    D, G = get_models(num_channels, d_conv_dim, g_conv_dim, z_dim, num_gpu,
                      d_model_restore, g_model_restore)

    # init optimizer
    criterion = nn.BCELoss()
    if torch.cuda.is_available():
        criterion.cuda()
    d_optimizer = optim.Adam(
        D.parameters(), lr=d_learning_rate, betas=(beta1, beta2))
    g_optimizer = optim.Adam(
        G.parameters(), lr=g_learning_rate, betas=(beta1, beta2))

    ###############
    # 2. training #
    ###############
    fixed_noise = make_variable(torch.randn(
        batch_size, z_dim, 1, 1).normal_(0, 1))

    for epoch in range(num_epochs):
        for step, (images, _) in enumerate(data_loader):
            batch_size = images.size(0)
            images = make_variable(images)
            real_labels = make_variable(torch.ones(batch_size))
            fake_labels = make_variable(torch.zeros(batch_size))

            ##############################
            # (1) training discriminator #
            ##############################
            for d_step in range(d_steps):
                d_optimizer.zero_grad()

                noise = make_variable(torch.randn(
                    batch_size, z_dim, 1, 1).normal_(0, 1))

                d_pred_real = D(images)
                d_loss_real = criterion(d_pred_real, real_labels)

                fake_images = G(noise)
                # use detach to avoid bp through G and spped up inference
                d_pred_fake = D(fake_images.detach())
                d_loss_fake = criterion(d_pred_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()

            ##########################
            # (2) training generator #
            ##########################
            for g_step in range(g_steps):
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()

                noise = make_variable(torch.randn(
                    batch_size, z_dim, 1, 1).normal_(0, 1))

                fake_images = G(noise)
                d_pred_fake = D(fake_images)
                g_loss = criterion(d_pred_fake, real_labels)
                g_loss.backward()

                g_optimizer.step()

            ##################
            # (3) print info #
            ##################
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

            ########################
            # (4) save fake images #
            ########################
            if ((step + 1) % sample_step == 0):
                if not os.path.exists(data_root):
                    os.makedirs(data_root)
                fake_images = G(fixed_noise)
                torchvision.utils.save_image(denormalize(fake_images.data),
                                             os.path.join(
                                                 data_root,
                                                 "DCGAN-fake-{}-{}.png"
                                                 .format(epoch + 1, step + 1))
                                             )

        #############################
        # (5) save model parameters #
        #############################
        if ((epoch + 1) % save_step == 0):
            if not os.path.exists(model_root):
                os.makedirs(model_root)
            torch.save(D.state_dict(), os.path.join(
                model_root, "DCGAN-discriminator-{}.pkl".format(epoch + 1)))
            torch.save(G.state_dict(), os.path.join(
                model_root, "DCGAN-generator-{}.pkl".format(epoch + 1)))
