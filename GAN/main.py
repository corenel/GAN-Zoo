#!/usr/bin/env python3
"""Main code for GAN."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from data_loader import mnist_dataloader
from models import Discriminator, Generator
from params import *
from utils import denormalize, make_variable

if __name__ == '__main__':
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

    # training
    for epoch in range(num_epochs):
        for idx, (images, _) in enumerate(mnist_dataloader):
            # convert tensor to variable
            images = make_variable(images.view(batch_size, -1))
            real_labels = make_variable(torch.ones(batch_size, 1))
            fake_labels = make_variable(torch.zeros(batch_size, 1))

            # training discriminator
            for d_idx in range(d_steps):
                d_optimizer.zero_grad()

                real_pred = D(images)
                d_real_loss = criterion(real_pred, real_labels)
                d_real_loss.backward()

                z = make_variable(torch.randn(batch_size, g_input_size))
                fake_images = G(z)
                fake_pred = D(fake_images)
                d_fake_loss = criterion(fake_pred, fake_labels)
                d_fake_loss.backward()

                d_optimizer.step()

                if ((idx + 1) % print_interval == 0):
                    print("Epoch [{}/{}] Step [{}/{}] D_STEP[{}/{}]:"
                          "d_loss={} D(x)={} D(G(z))={}"
                          .format(epoch,
                                  num_epochs,
                                  idx + 1,
                                  len(mnist_dataloader),
                                  d_idx + 1,
                                  d_steps,
                                  d_real_loss.data[0] + d_fake_loss.data[0],
                                  d_real_loss.data[0],
                                  d_fake_loss.data[0]))

            # training generator
            for g_idx in range(g_steps):
                D.zero_grad()
                G.zero_grad()

                z = make_variable(torch.randn(batch_size, g_input_size))
                fake_images = G(z)
                fake_pred = D(fake_images)
                # note that we use real_labels there
                g_fake_loss = criterion(fake_pred, real_labels)
                g_fake_loss.backward()

                g_optimizer.step()

                if ((idx + 1) % print_interval == 0):
                    print("Epoch [{}/{}] Step [{}/{}] G_STEP[{}/{}]:"
                          "g_loss={}"
                          .format(epoch,
                                  num_epochs,
                                  idx + 1,
                                  len(mnist_dataloader),
                                  g_idx + 1,
                                  g_steps,
                                  g_fake_loss.data[0]))

        fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
        torchvision.utils.save_image(denormalize(fake_images.data),
                                     "../data/GAN_fake_images-{}.png"
                                     .format(epoch + 1))

    # Save the trained parameters
    torch.save(G.state_dict(), './generator.pkl')
    torch.save(D.state_dict(), './discriminator.pkl')
