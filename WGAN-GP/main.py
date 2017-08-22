#!/usr/bin/env python3
"""Main code for WGAN-GP."""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import params
from data_loader import data_loader, get_data_iterator
from models import get_models
from utils import (calc_gradient_penalty, init_random_seed, make_variable,
                   save_fake_image, save_model)

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
        params.batch_size, params.z_dim, 1, 1).normal_(0, 1), volatile=True)
    data_step = 0
    data_iter = get_data_iterator()

    for epoch in range(params.num_epochs):
        ##############################
        # (1) training discriminator #
        ##############################
        # requires to compute gradients for D
        for p in D.parameters():
            p.requires_grad = True

        # set steps for discriminator
        if g_step_counter < 25 or g_step_counter % 500 == 0:
            # this helps to start with the critic at optimum
            # even in the first iterations.
            critic_iters = 100
        else:
            critic_iters = params.d_steps

        # loop for optimizing discriminator
        for d_step in range(critic_iters):
            # make images torch.Variable
            images = next(data_iter)
            data_step += 1
            images = make_variable(images)
            if images.size(0) != params.batch_size:
                continue

            # zero grad for optimizer of discriminator
            d_optimizer.zero_grad()

            # compute real data loss for discriminator
            d_loss_real = D(images)
            d_loss_real = d_loss_real.mean()
            d_loss_real.backward(fake_labels)

            # compute fake data loss for discriminator
            noise = make_variable(torch.randn(
                params.batch_size, params.z_dim, 1, 1).normal_(0, 1),
                volatile=True)
            fake_images = make_variable(G(noise).data)
            d_loss_fake = D(fake_images.detach())
            d_loss_fake = d_loss_fake.mean()
            d_loss_fake.backward(real_labels)

            # compute gradient penalty
            gradient_penalty = calc_gradient_penalty(
                D, images.data, fake_images.data)
            gradient_penalty.backward()

            # optimize weights of discriminator
            d_loss = - d_loss_real + d_loss_fake + gradient_penalty
            d_optimizer.step()

        ##########################
        # (2) training generator #
        ##########################
        # avoid to compute gradients for D
        for p in D.parameters():
            p.requires_grad = False

        # zero grad for optimizer of generator
        g_optimizer.zero_grad()

        # generate fake images
        noise = make_variable(torch.randn(
            params.batch_size, params.z_dim, 1, 1).normal_(0, 1))
        fake_images = G(noise)

        # compute loss for generator
        g_loss = D(fake_images).mean()
        g_loss.backward(fake_labels)
        g_loss = - g_loss

        # optimize weights of generator
        g_optimizer.step()
        g_step_counter += 1

        ##################
        # (3) print info #
        ##################
        if ((g_step_counter + 1) % params.log_step == 0):
            print("Epoch [{}/{}] Step [{}/{}] G_STEP[{}]:"
                  "d_loss={:.5f} g_loss={:.5f} "
                  "D(x)={:.5f} D(G(z))={:.5f} GP={:.5f}"
                  .format(epoch + 1,
                          params.num_epochs,
                          (data_step + 1) % len(data_loader),
                          len(data_loader),
                          g_step_counter + 1,
                          d_loss.data[0],
                          g_loss.data[0],
                          d_loss_real.data[0],
                          d_loss_fake.data[0],
                          gradient_penalty.data[0])
                  )

        ########################
        # (4) save fake images #
        ########################
        if ((g_step_counter + 1) % params.sample_step == 0):
            save_fake_image(
                G, fixed_noise,
                "WGAN-GP_fake_image-{}.png".format(g_step_counter + 1))

    #############################
    # (5) save model parameters #
    #############################
    if ((epoch + 1) % params.save_step == 0):
        save_model(D, "WGAN-GP_discriminator-{}.pt".format(epoch + 1))
        save_model(G, "WGAN-GP_generator-{}.pt".format(epoch + 1))
