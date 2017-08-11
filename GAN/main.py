#!/usr/bin/env python3
"""Main code for GAN."""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from data_loader import mnist_dataloader
from models import Discriminator, Generator
from params import *
from utils import denormalize, make_variable

if __name__ == '__main__':
    ####################
    # 1. setup network #
    ####################

    # init models
    D = Discriminator(input_size=d_input_size,
                      hidden_size=d_hidden_size,
                      output_size=d_output_size)
    G = Generator(input_size=g_input_size,
                  hidden_size=g_hidden_size,
                  output_size=g_output_size)

    # check if cuda is available
    if torch.cuda.is_available():
        D.cuda()
        G.cuda()

    # init criterion and optimizer
    criterion = nn.BCELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)

    ###############
    # 2. training #
    ###############
    for epoch in range(num_epochs):
        for step, (images, _) in enumerate(mnist_dataloader):
            # convert tensor to variable
            images = make_variable(images.view(batch_size, -1))
            real_labels = make_variable(torch.ones(batch_size, 1))
            fake_labels = make_variable(torch.zeros(batch_size, 1))

            ##############################
            # (1) training discriminator #
            ##############################
            for d_idx in range(d_steps):
                d_optimizer.zero_grad()

                d_pred_real = D(images)
                d_loss_real = criterion(d_pred_real, real_labels)
                d_loss_real.backward()

                z = make_variable(torch.randn(batch_size, g_input_size))
                fake_images = G(z)
                d_pred_fake = D(fake_images)
                d_loss_fake = criterion(d_pred_fake, fake_labels)
                d_loss_fake.backward()

                d_optimizer.step()

                if ((step + 1) % log_step == 0):
                    print("Epoch [{}/{}] Step [{}/{}] D_STEP[{}/{}]:"
                          "d_loss={} D(x)={} D(G(z))={}"
                          .format(epoch,
                                  num_epochs,
                                  step + 1,
                                  len(mnist_dataloader),
                                  d_idx + 1,
                                  d_steps,
                                  d_loss_real.data[0] + d_loss_fake.data[0],
                                  d_loss_real.data[0],
                                  d_loss_fake.data[0]))

            ##########################
            # (2) training generator #
            ##########################
            for g_idx in range(g_steps):
                D.zero_grad()
                G.zero_grad()

                z = make_variable(torch.randn(batch_size, g_input_size))
                fake_images = G(z)
                d_pred_fake = D(fake_images)
                # note that we use real_labels there
                g_fake_loss = criterion(d_pred_fake, real_labels)
                g_fake_loss.backward()

                g_optimizer.step()

                if ((step + 1) % log_step == 0):
                    print("Epoch [{}/{}] Step [{}/{}] G_STEP[{}/{}]:"
                          "g_loss={}"
                          .format(epoch,
                                  num_epochs,
                                  step + 1,
                                  len(mnist_dataloader),
                                  g_idx + 1,
                                  g_steps,
                                  g_fake_loss.data[0]))

            ########################
            # (3) save fake images #
            ########################
            if ((step + 1) % sample_step == 0):
                if not os.path.exists(data_root):
                    os.makedirs(data_root)
                fake_images = fake_images.view(
                    fake_images.size(0), 1, image_size, image_size)
                torchvision.utils.save_image(denormalize(fake_images.data),
                                             os.path.join(
                                                 data_root,
                                                 "GAN-fake-{}-{}.png"
                                                 .format(epoch + 1, step + 1))
                                             )

        #############################
        # (4) save model parameters #
        #############################
        if ((epoch + 1) % save_step == 0):
            if not os.path.exists(model_root):
                os.makedirs(model_root)
            torch.save(D.state_dict(), os.path.join(
                model_root, "GAN-discriminator-{}.pkl".format(epoch + 1)))
            torch.save(G.state_dict(), os.path.join(
                model_root, "GAN-generator-{}.pkl".format(epoch + 1)))
