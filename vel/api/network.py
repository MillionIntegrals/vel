import torch.nn as nn

from .size_hint import SizeHints


class Network(nn.Module):
    """ Vel wrapper over nn.Module offering a few internally useful utilities """

    def reset_weights(self):
        """ Call proper initializers for the weights """
        pass

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return False

    def zero_state(self, batch_size):
        """ Potential state for the model """
        return None


class BackboneNetwork(Network):
    """ Network, whose output feeds into other models. Needs to provide size hints. """

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        raise NotImplementedError
