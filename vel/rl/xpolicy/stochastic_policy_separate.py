import gym
import typing

from vel.api import LinearBackboneModel, ModelFactory, BackboneModel
from vel.module.input.identity import IdentityFactory
from vel.rl.module.stochastic_action_head import StochasticActionHead
from vel.rl.module.value_head import ValueHead


class StochasticPolicyModelSeparate(BackboneModel):
    """
    Policy gradient model class with an actor and critic heads that don't share a backbone
    """

    def __init__(self, input_block: BackboneModel,
                 policy_backbone: LinearBackboneModel, value_backbone: LinearBackboneModel,
                 action_space: gym.Space):
        super().__init__()

        self.input_block = input_block
        self.policy_backbone = policy_backbone
        self.value_backbone = value_backbone

        self.action_head = StochasticActionHead(
            action_space=action_space,
            input_dim=self.policy_backbone.output_dim
        )

        self.value_head = ValueHead(input_dim=self.value_backbone.output_dim)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.input_block.reset_weights()

        self.policy_backbone.reset_weights()
        self.value_backbone.reset_weights()

        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observations):
        """ Calculate model outputs """
        input_data = self.input_block(observations)

        policy_base_output = self.policy_backbone(input_data)
        value_base_output = self.value_backbone(input_data)

        action_output = self.action_head(policy_base_output)
        value_output = self.value_head(value_base_output)

        return action_output, value_output

    def value(self, observations, state=None):
        """ Calculate only value head for given state """
        input_data = self.input_block(observations)
        base_output = self.value_backbone(input_data)
        value_output = self.value_head(base_output)
        return value_output

    def policy(self, observations):
        """ Calculate only action head for given state """
        input_data = self.input_block(observations)
        policy_base_output = self.policy_backbone(input_data)
        policy_params = self.action_head(policy_base_output)
        return policy_params


class StochasticPolicyModelSeparateFactory(ModelFactory):
    """ Factory class for policy gradient models """
    def __init__(self, input_block: ModelFactory, policy_backbone: ModelFactory, value_backbone: ModelFactory):
        self.input_block = input_block
        self.policy_backbone = policy_backbone
        self.value_backbone = value_backbone

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        input_block = self.input_block.instantiate()
        policy_backbone = self.policy_backbone.instantiate(**extra_args)
        value_backbone = self.value_backbone.instantiate(**extra_args)

        return StochasticPolicyModelSeparate(input_block, policy_backbone, value_backbone, extra_args['action_space'])


def create(policy_backbone: ModelFactory, value_backbone: ModelFactory,
           input_block: typing.Optional[ModelFactory] = None):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return StochasticPolicyModelSeparateFactory(
        input_block=input_block,
        policy_backbone=policy_backbone,
        value_backbone=value_backbone
    )
