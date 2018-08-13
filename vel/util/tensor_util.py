import torch


def one_hot_encoding(input_tensor, num_labels):
    """ One-hot encode labels from input """
    return torch.eye(num_labels, device=input_tensor.device)[input_tensor]
