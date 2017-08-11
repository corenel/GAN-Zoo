"""Params for WGAN-GP."""

# params for dataset
data_root = "../data/"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 64
image_size = 64

# params for model
model_root = "../model/"
num_channels = 3
z_dim = 100
d_conv_dim = 64
g_conv_dim = 64
d_model_restore = None
g_model_restore = None

# params for training
num_gpu = 1
num_epochs = 25
log_step = 10
sample_step = 100
save_step = 10
manual_seed = None
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999
