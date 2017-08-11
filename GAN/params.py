"""Params for GAN."""

# params for MNIST dataset
data_root = "../data/"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 100
image_size = 28

# params for discriminator model
d_input_size = image_size ** 2  # 28*28
d_hidden_size = 256
d_output_size = 1
d_learning_rate = 0.0003
d_steps = 1

# params for generator model
g_input_size = 64
g_hidden_size = 256
g_output_size = image_size ** 2  # 28*28
g_learning_rate = 0.0003
g_steps = 1

# params for training
model_root = "../model/"
num_epochs = 200
log_step = 100
sample_step = 100
save_step = 10
