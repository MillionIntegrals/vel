import gym
import typing

from vel.api import LinearBackboneModel, ModelFactory, BackboneModel
from vel.module.input.identity import IdentityFactory
from vel.rl.module.stochastic_action_head import StochasticActionHead
from vel.rl.module.value_head import ValueHead


class StochasticPolicy(BackboneModel):
    """
    Most generic policy gradient model class with a set of common actor-critic heads that share a single backbone
    """

    def __init__(self, input_block: BackboneModel, backbone: LinearBackboneModel, action_space: gym.Space):
        super().__init__()

        self.input_block = input_block
        self.backbone = backbone

        assert not self.backbone.is_stateful, "Backbone shouldn't have state"

        self.action_head = StochasticActionHead(
            action_space=action_space,
            input_dim=self.backbone.output_dim
        )

        self.value_head = ValueHead(
            input_dim=self.backbone.output_dim
        )

    def reset_weights(self):
        """ Initialize properly model weights """
        self.input_block.reset_weights()
        self.backbone.reset_weights()
        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observation):
        """ Calculate model outputs """
        input_data = self.input_block(observation)

        base_output = self.backbone(input_data)

        action_output = self.action_head(base_output)
        value_output = self.value_head(base_output)

        return action_output, value_output


class StochasticPolicyFactory(ModelFactory):
    """ Factory class for policy gradient models """
    def __init__(self, input_block: IdentityFactory, backbone: ModelFactory):
        self.backbone = backbone
        self.input_block = input_block

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        input_block = self.input_block.instantiate()
        backbone = self.backbone.instantiate(**extra_args)

        return StochasticPolicy(input_block, backbone, extra_args['action_space'])


def create(backbone: ModelFactory, input_block: typing.Optional[ModelFactory] = None):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return StochasticPolicyFactory(input_block=input_block, backbone=backbone)
