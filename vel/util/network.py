import torch.nn as nn


ACTIVATION_DICT = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'elu': nn.ELU,
    'leaky_relu': nn.LeakyReLU
}

NORMALIZATION_DICT = {
    'layer': nn.LayerNorm,
    'layer-noscale': lambda normalized_shape: nn.LayerNorm(normalized_shape, elementwise_affine=False),
    'batch1d': nn.BatchNorm1d,
    'batch2d': nn.BatchNorm2d
}


def activation(name):
    """ Return activation block corresponding given name """
    return ACTIVATION_DICT[name]


def normalization(name):
    """ Return activation block corresponding given name """
    return NORMALIZATION_DICT[name]


def convolution_size_equation(size, filter_size, padding, stride):
    """ Output size of convolutional layer """
    return (size - filter_size + 2 * padding) // stride + 1


def convolutional_layer_series(initial_size, layer_sequence):
    """ Execute a series of convolutional layer transformations to the size number """
    size = initial_size

    for filter_size, padding, stride in layer_sequence:
        size = convolution_size_equation(size, filter_size, padding, stride)

    return size
