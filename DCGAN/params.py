"""Params for DCGAN."""

# params for dataset
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 128
image_size = 64

# params for model
num_channels = 3
z_dim = 100
d_conv_dim = 64
g_conv_dim = 64
bn_beta1 = 0.5
bn_beta2 = 0.999

# params for training
num_workers = 0
num_epochs = 200
print_interval = 8
manual_seed = None
learning_rate = 0.0002
