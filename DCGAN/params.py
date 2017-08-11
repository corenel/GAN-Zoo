"""Params for DCGAN."""

# params for dataset
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 100

# params for training
num_workers = 0
num_epochs = 30000
print_interval = 10
