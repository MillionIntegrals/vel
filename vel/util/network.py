

def convolution_size_equation(size, filter_size, padding, stride):
    """ Output size of convolutional layer """
    return (size - filter_size + 2 * padding) // stride + 1


def convolutional_layer_series(initial_size, layer_sequence):
    """ Execute a series of convolutional layer transformations to the size number """
    size = initial_size

    for filter_size, padding, stride in layer_sequence:
        size = convolution_size_equation(size, filter_size, padding, stride)

    return size