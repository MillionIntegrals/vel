import itertools as it

import torch
import torch.nn as nn
import torch.nn.functional as F

from vel.api import (
    LossFunctionModel, ModuleFactory, VModule, BackboneModule, SizeHints, SizeHint, OptimizerFactory,
    VelOptimizer
)
from vel.metric.accuracy import Accuracy
from vel.metric.loss_metric import Loss


class SequenceClassification(LossFunctionModel):
    """ NLP (text) sequence classification """

    def __init__(self, net: BackboneModule, output_size: int):
        super().__init__()

        self.net = net
        self.output_layer = nn.Linear(
            in_features=self.net.size_hints().assert_single().last(),
            out_features=output_size
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
            output = F.log_softmax(self.output_layer(output), dim=-1)
            return output, new_state
        else:
            output = self.net(input_data)
            output = F.log_softmax(self.output_layer(output), dim=-1)
            return output

    def loss_value(self, x_data, y_true, y_pred) -> torch.tensor:
        """ Calculate a value of loss function """
        return F.nll_loss(y_pred, y_true)

    def create_optimizer(self, optimizer_factory: OptimizerFactory) -> VelOptimizer:
        grouped = self.net.grouped_parameters()
        parameters = it.chain(grouped, [("output", self.output_layer.parameters())])
        return optimizer_factory.instantiate(parameters)

    def metrics(self) -> list:
        """ Set of metrics for this model """
        return [Loss(), Accuracy()]


class SequenceClassificationFactory(ModuleFactory):
    def __init__(self, net_factory: ModuleFactory, alphabet_size: int, output_dim: int):
        self.net_factory = net_factory
        self.output_dim = output_dim
        self.alphabet_size = alphabet_size

    def instantiate(self, **extra_args) -> VModule:
        size_hint = SizeHints(SizeHint(None, None))
        net = self.net_factory.instantiate(alphabet_size=self.alphabet_size, size_hint=size_hint)

        return SequenceClassification(
            net=net, output_size=self.output_dim
        )


def create(loader, net: ModuleFactory, output_dim: int):
    """ Vel factory function """
    return SequenceClassificationFactory(
        net_factory=net,
        alphabet_size=loader.alphabet_size,
        output_dim=output_dim
    )
