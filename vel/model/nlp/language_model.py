import torch
import torch.nn as nn
import torch.nn.functional as F

from vel.api import LossFunctionModel, ModelFactory, Network, BackboneNetwork, SizeHints, SizeHint


class LanguageModel(LossFunctionModel):
    """ Language model - autoregressive generative model for text """

    def __init__(self, alphabet_size: int, net: BackboneNetwork):
        super().__init__()

        self.net = net
        self.alphabet_size = alphabet_size
        self.output_dim = self.alphabet_size + 1

        self.net = net
        self.output_layer = nn.Linear(
            in_features=self.net.size_hints().assert_single().last(),
            out_features=self.alphabet_size+1
        )

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return self.net.is_stateful

    def zero_state(self, batch_size):
        """ Potential state for the model """
        return self.net.zero_state(batch_size)

    def forward(self, input_data: torch.Tensor, state=None) -> torch.Tensor:
        r"""Defines the computation performed at every call.

        Should be overridden by all subclasses.

        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.
        """
        if self.net.is_stateful:
            output, new_state = self.net(input_data, state=state)
        else:
            output = self.net(input_data)
            new_state = state

        return F.log_softmax(self.output_layer(output), dim=-1), new_state

    def loss_value(self, x_data, y_true, y_pred) -> torch.tensor:
        """ Calculate a value of loss function """
        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1).to(torch.long)
        return F.nll_loss(y_pred, y_true)


class LanguageModelFactory(ModelFactory):
    def __init__(self, alphabet_size: int, net_factory: ModelFactory):
        self.alphabet_size = alphabet_size
        self.net_factory = net_factory

    def instantiate(self, **extra_args) -> Network:
        size_hint = SizeHints(SizeHint(None, None))
        net = self.net_factory.instantiate(alphabet_size=self.alphabet_size, size_hint=size_hint)

        return LanguageModel(
            alphabet_size=self.alphabet_size,
            net=net
        )


def create(loader, net: ModelFactory):
    """ Vel factory function """
    return LanguageModelFactory(
        alphabet_size=loader.alphabet_size,
        net_factory=net
    )
