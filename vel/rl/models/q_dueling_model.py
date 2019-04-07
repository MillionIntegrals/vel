import gym
import typing

from vel.api import LinearBackboneModel, Model, ModelFactory, BackboneModel
from vel.modules.input.identity import IdentityFactory
from vel.rl.api import Rollout, Evaluator
from vel.rl.modules.q_dueling_head import QDuelingHead
from vel.rl.models.q_model import QModelEvaluator


class QDuelingModel(Model):
    """
    Deterministic greedy action-value model with dueling heads (kind of actor and critic)
    Supports only discrete action spaces (ones that can be enumerated)
    """

    def __init__(self, input_block: BackboneModel, backbone: LinearBackboneModel, action_space: gym.Space):
        super().__init__()

        self.action_space = action_space

        self.input_block = input_block
        self.backbone = backbone
        self.q_head = QDuelingHead(input_dim=backbone.output_dim, action_space=action_space)

    def forward(self, observations):
        """ Model forward pass """
        observations = self.input_block(observations)
        advantage_features, value_features = self.backbone(observations)
        q_values = self.q_head(advantage_features, value_features)

        return q_values

    def reset_weights(self):
        """ Initialize weights to reasonable defaults """
        self.input_block.reset_weights()
        self.backbone.reset_weights()
        self.q_head.reset_weights()

    def step(self, observations):
        """ Sample action from an action space for given state """
        q_values = self(observations)

        return {
            'actions': self.q_head.sample(q_values),
            'q': q_values
        }

    def evaluate(self, rollout: Rollout) -> Evaluator:
        """ Evaluate model on a rollout """
        return QModelEvaluator(self, rollout)


class QDuelingModelFactory(ModelFactory):
    """ Factory class for policy gradient models """
    def __init__(self, input_block: ModelFactory, backbone: ModelFactory):
        self.input_block = input_block
        self.backbone = backbone

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        input_block = self.input_block.instantiate()
        backbone = self.backbone.instantiate(**extra_args)

        return QDuelingModel(input_block, backbone, extra_args['action_space'])


def create(backbone: ModelFactory, input_block: typing.Optional[ModelFactory]=None):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return QDuelingModelFactory(input_block=input_block, backbone=backbone)
