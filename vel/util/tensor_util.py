import torch


def one_hot_encoding(input_tensor, num_labels):
    """ One-hot encode labels from input """
    return torch.eye(num_labels, device=input_tensor.device)[input_tensor.to(torch.long)]


def merge_first_two_dims(tensor):
    """ Reshape tensor to merge first two dimensions """
    shape = tensor.shape
    batch_size = shape[0] * shape[1]
    new_shape = tuple([batch_size] + list(shape[2:]))
    return tensor.view(new_shape)
