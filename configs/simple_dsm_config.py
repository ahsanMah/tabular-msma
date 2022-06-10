import ml_collections

"""
Configuration for vanilla (un-conditioned model) denoising score matching  
"""


def get_configs():
    config = ml_collections.ConfigDict()

    config.training = training = ml_collections.ConfigDict()
    training.scale = 0.5
    training.reduce_op = "sum"
    # training.hidden_size = 64

    config.data = data = ml_collections.ConfigDict()
    data.dataset = "credit_fraud"
    data.input_dims = 29

    return config
