#!/usr/bin/env python3
"""Main code for WGAN-GP."""

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
    D, G = get_models()

    # init optimizer
    if use_Adam:
        d_optimizer = optim.Adam(
            D.parameters(), lr=d_learning_rate, betas=(beta1, beta2))
        g_optimizer = optim.Adam(
            G.parameters(), lr=g_learning_rate, betas=(beta1, beta2))
    else:
        d_optimizer = optim.RMSprop(D.parameters(), lr=d_learning_rate)
        g_optimizer = optim.RMSprop(G.parameters(), lr=g_learning_rate)

    ###############
    # 2. training #
    ###############
    g_step_counter = 0
    fixed_noise = make_variable(torch.randn(
        batch_size, z_dim, 1, 1).normal_(0, 1))

    for epoch in range(num_epochs):
        step = 0
        data_iter = iter(data_loader)
        while step < len(data_loader):
            ##############################
            # (1) training discriminator #
            ##############################

            # set steps for discriminator
            if g_step_counter < 25 or g_step_counter % 500 == 0:
                critic_iters = 100
            else:
                critic_iters = d_steps

            # loop for optimizing discriminator
            for d_step in range(critic_iters):
                # clamp gradient value
                for p in D.parameters():
                    p.data.clamp_(clamp_lower, clamp_upper)

                if step < len(data_loader):
                    images, _ = next(data_iter)
                    images = make_variable(images)
                    batch_size = images.size(0)
                    real_labels = make_variable(torch.ones(batch_size))
                    fake_labels = make_variable(torch.zeros(batch_size))
                    step += 1
                else:
                    break

                d_optimizer.zero_grad()

                noise = make_variable(torch.randn(
                    batch_size, z_dim, 1, 1).normal_(0, 1))

                d_loss_real = D(images)
                d_loss_real.backward(real_labels)

                fake_images = G(noise)
                # use detach to avoid bp through G and spped up inference
                d_loss_fake = D(fake_images.detach())
                d_loss_fake.backward(fake_labels)

                d_loss = d_loss_real - d_loss_fake
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
                g_loss = D(fake_images)
                g_loss.backward(real_labels)

                g_optimizer.step()
                g_step_counter += 1

            ##################
            # (3) print info #
            ##################
            if ((g_step_counter + 1) % log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}] G_STEP[{}]:"
                      "d_loss={} g_loss={} D(x)={} D(G(z))={}"
                      .format(epoch + 1,
                              num_epochs,
                              step + 1,
                              len(data_loader),
                              g_step_counter + 1,
                              d_loss.data[0],
                              g_loss.data[0],
                              d_loss_real.data[0],
                              d_loss_fake.data[0]))

            ########################
            # (4) save fake images #
            ########################
            if ((g_step_counter + 1) % sample_step == 0):
                if not os.path.exists(data_root):
                    os.makedirs(data_root)
                fake_images = G(fixed_noise)
                torchvision.utils.save_image(denormalize(fake_images.data),
                                             os.path.join(
                                                 data_root,
                                                 "WGAN-fake-{}-{}.png"
                                                 .format(epoch + 1, step + 1))
                                             )

        #############################
        # (5) save model parameters #
        #############################
        if ((epoch + 1) % save_step == 0):
            if not os.path.exists(model_root):
                os.makedirs(model_root)
            torch.save(D.state_dict(), os.path.join(
                model_root, "WGAN-discriminator-{}.pkl".format(epoch + 1)))
            torch.save(G.state_dict(), os.path.join(
                model_root, "WGAN-generator-{}.pkl".format(epoch + 1)))
