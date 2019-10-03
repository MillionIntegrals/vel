import torch
import collections


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


def to_device(tensor, device: torch.device):
    """ Convert tensor-like object to given PyTorch device """
    if tensor is None:
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, dict):
        return {k: to_device(v, device) for k, v in tensor.items()}
    elif isinstance(tensor, list):
        return [to_device(v, device) for v in tensor]
    elif isinstance(tensor, tuple):
        return tuple(to_device(v, device) for v in tensor)
    else:
        raise NotImplementedError


class TensorAccumulator:
    """ Buffer for tensors that will be stacked together """
    def __init__(self):
        self.accumulants = collections.defaultdict(list)

    def add(self, name, tensor):
        self.accumulants[name].append(tensor)

    def result(self):
        """ Concatenate accumulated tensors """
        return {k: torch.stack(v) for k, v in self.accumulants.items()}
