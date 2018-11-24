import torch


def one_hot_encoding(input_tensor, num_labels):
    """ One-hot encode labels from input """
    xview = input_tensor.view(-1, 1).to(torch.long)

    onehot = torch.zeros(xview.size(0), num_labels, device=input_tensor.device, dtype=torch.float)
    onehot.scatter_(1, xview, 1)
    return onehot.view(list(input_tensor.shape) + [-1])


def merge_first_two_dims(tensor):
    """ Reshape tensor to merge first two dimensions """
    shape = tensor.shape
    batch_size = shape[0] * shape[1]
    new_shape = tuple([batch_size] + list(shape[2:]))
    return tensor.view(new_shape)
