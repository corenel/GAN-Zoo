#!/usr/bin/env python3
"""Main code for WGAN-GP."""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import params
from data_loader import data_loader
from models import get_models
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
    if params.use_Adam:
        d_optimizer = optim.Adam(
            D.parameters(), lr=params.d_learning_rate, betas=(params.beta1,
                                                              params.beta2))
        g_optimizer = optim.Adam(
            G.parameters(), lr=params.g_learning_rate, betas=(params.beta1,
                                                              params.beta2))
    else:
        d_optimizer = optim.RMSprop(D.parameters(), lr=params.d_learning_rate)
        g_optimizer = optim.RMSprop(G.parameters(), lr=params.g_learning_rate)

    ###############
    # 2. training #
    ###############
    g_step_counter = 0
    real_labels = make_variable(torch.FloatTensor([1]))
    fake_labels = make_variable(torch.FloatTensor([-1]))
    fixed_noise = make_variable(torch.randn(
        params.batch_size, params.z_dim, 1, 1).normal_(0, 1))

    for epoch in range(params.num_epochs):
        data_step = 0
        data_iter = iter(data_loader)
        while data_step < len(data_loader):
            ##############################
            # (1) training discriminator #
            ##############################
            # requires to compute gradients for D
            for p in D.parameters():
                p.requires_grad = True

            # set steps for discriminator
            if g_step_counter < 25 or g_step_counter % 500 == 0:
                critic_iters = 100
            else:
                critic_iters = params.d_steps

            # loop for optimizing discriminator
            for d_step in range(critic_iters):

                if data_step < len(data_loader):
                    images, _ = next(data_iter)
                    images = make_variable(images)
                    # batch_size = images.size(0)
                    data_step += 1
                else:
                    break

                d_optimizer.zero_grad()

                d_loss_real = D(images)
                d_loss_real.backward(real_labels)

                noise = make_variable(torch.randn(
                    params.batch_size, params.z_dim, 1, 1).normal_(0, 1),
                    volatile=True)
                fake_images = make_variable(G(noise).data)
                d_loss_fake = D(fake_images)
                d_loss_fake.backward(fake_labels)

                d_loss = d_loss_real - d_loss_fake
                d_optimizer.step()

                # clamp gradient value
                for p in D.parameters():
                    p.data.clamp_(params.clamp_lower, params.clamp_upper)

            ##########################
            # (2) training generator #
            ##########################
            # avoid to compute gradients for D
            for p in D.parameters():
                p.requires_grad = False  # to avoid computation

            for g_step in range(params.g_steps):
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()

                noise = make_variable(torch.randn(
                    params.batch_size, params.z_dim, 1, 1).normal_(0, 1))

                fake_images = G(noise)
                g_loss = D(fake_images)
                g_loss.backward(real_labels)

                g_optimizer.step()
                g_step_counter += 1

            ##################
            # (3) print info #
            ##################
            if ((g_step_counter + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}] G_STEP[{}]:"
                      "d_loss={} g_loss={} D(x)={} D(G(z))={}"
                      .format(epoch + 1,
                              params.num_epochs,
                              data_step + 1,
                              len(data_loader),
                              g_step_counter + 1,
                              d_loss.data[0],
                              g_loss.data[0],
                              d_loss_real.data[0],
                              d_loss_fake.data[0]))

            ########################
            # (4) save fake images #
            ########################
            if ((g_step_counter + 1) % params.sample_step == 0):
                if not os.path.exists(params.data_root):
                    os.makedirs(params.data_root)
                fake_images = G(fixed_noise)
                torchvision.utils.save_image(denormalize(fake_images.data),
                                             os.path.join(
                                                 params.data_root,
                                                 "WGAN-fake-{}-{}.png"
                                                 .format(epoch + 1,
                                                         data_step + 1))
                                             )

        #############################
        # (5) save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):
            if not os.path.exists(params.model_root):
                os.makedirs(params.model_root)
            torch.save(D.state_dict(), os.path.join(
                params.model_root,
                "WGAN-discriminator-{}.pkl".format(epoch + 1)))
            torch.save(G.state_dict(), os.path.join(
                params.model_root,
                "WGAN-generator-{}.pkl".format(epoch + 1)))
