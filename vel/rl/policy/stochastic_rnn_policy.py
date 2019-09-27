import gym
import typing

from vel.api import LinearBackboneModel, ModelFactory, BackboneModel
from vel.module.input.identity import IdentityFactory
from vel.rl.module.stochastic_action_head import StochasticActionHead
from vel.rl.module.value_head import ValueHead


class StochasticRnnPolicy(BackboneModel):
    """
    Most generic policy gradient model class with a set of common actor-critic heads that share a single backbone
    RNN version
    """

    def __init__(self, input_block: BackboneModel, backbone: LinearBackboneModel,
                 action_space: gym.Space):
        super().__init__()

        self.input_block = input_block
        self.backbone = backbone

        assert self.backbone.is_stateful, "Must have a stateful backbone"

        self.action_head = StochasticActionHead(
            action_space=action_space,
            input_dim=self.backbone.output_dim
        )
        self.value_head = ValueHead(input_dim=self.backbone.output_dim)

        assert self.backbone.is_stateful, "Backbone must be a recurrent model"

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return True

    def zero_state(self, batch_size):
        return self.backbone.zero_state(batch_size)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.input_block.reset_weights()
        self.backbone.reset_weights()
        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observations, state):
        """ Calculate model outputs """
        input_data = self.input_block(observations)
        base_output, new_state = self.backbone(input_data, state=state)

        action_output = self.action_head(base_output)
        value_output = self.value_head(base_output)

        return action_output, value_output, new_state

    def value(self, observation, state=None):
        """ Calculate only value head for given state """
        input_data = self.input_block(observation)

        base_output, new_state = self.backbone(input_data, state)
        value_output = self.value_head(base_output)

        return value_output

    def reset_state(self, state, dones):
        """ Reset the state after the episode has been terminated """
        if (dones > 0).any().item():
            zero_state = self.backbone.zero_state(dones.shape[0]).to(state.device)
            dones_expanded = dones.unsqueeze(-1)
            return state * (1 - dones_expanded) + zero_state * dones_expanded
        else:
            return state


class StochasticRnnPolicyFactory(ModelFactory):
    """ Factory class for policy gradient models """
    def __init__(self, input_block: ModelFactory, backbone: ModelFactory):
        self.input_block = input_block
        self.backbone = backbone

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        input_block = self.input_block.instantiate()
        backbone = self.backbone.instantiate(**extra_args)

        return StochasticRnnPolicy(input_block, backbone, extra_args['action_space'])


def create(backbone: ModelFactory, input_block: typing.Optional[ModelFactory] = None):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return StochasticRnnPolicyFactory(
        input_block=input_block,
        backbone=backbone
    )
