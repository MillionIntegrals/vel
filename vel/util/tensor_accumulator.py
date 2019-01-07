import collections

import torch


class TensorAccumulator:
    """ Buffer for tensors that will be stacked together """
    def __init__(self):
        self.accumulants = collections.defaultdict(list)

    def add(self, name, tensor):
        self.accumulants[name].append(tensor)

    def result(self):
        """ Concatenate accumulated tensors """
        return {k: torch.stack(v) for k, v in self.accumulants.items()}
