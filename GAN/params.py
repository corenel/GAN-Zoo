"""Params for GAN."""

# params for MNIST dataset
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)
batch_size = 100

# params for discriminator model
d_input_size = 784  # 28*28
d_hidden_size = 256
d_output_size = 1
d_learning_rate = 0.0003
d_steps = 1

# params for generator model
g_input_size = 64
g_hidden_size = 256
g_output_size = 784  # 28*28
g_learning_rate = 0.0003
g_steps = 1

# params for training
num_epochs = 30000
print_interval = 100
