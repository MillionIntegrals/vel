import gym
import typing

from vel.api import LinearBackboneModel, ModelFactory, BackboneModel
from vel.modules.input.identity import IdentityFactory
from vel.rl.api import Rollout, RlModel, Evaluator
from vel.rl.modules.q_head import QHead


class QModelEvaluator(Evaluator):
    """ Evaluate simple q-model """
    def __init__(self, model: 'QModel', rollout: Rollout):
        super().__init__(rollout)
        self.model = model

    @Evaluator.provides('model:q')
    def model_q(self):
        """ Action values for all (discrete) actions """
        observations = self.get('rollout:observations')
        return self.model(observations)

    @Evaluator.provides('model:action:q')
    def model_action_q(self):
        """ Action values for selected actions in the rollout """
        q = self.get('model:q')
        actions = self.get('rollout:actions')
        return q.gather(1, actions.unsqueeze(1)).squeeze(1)

    @Evaluator.provides('model:q_next')
    def model_q_next(self):
        """ Action values for all (discrete) actions """
        observations = self.get('rollout:observations_next')
        return self.model(observations)


class QModel(RlModel):
    """
    Simple deterministic greedy action-value model.
    Supports only discrete action spaces (ones that can be enumerated)
    """
    def __init__(self, input_block: BackboneModel, backbone: LinearBackboneModel, action_space: gym.Space):
        super().__init__()

        self.action_space = action_space

        self.input_block = input_block
        self.backbone = backbone
        self.q_head = QHead(input_dim=backbone.output_dim, action_space=action_space)

    def reset_weights(self):
        """ Initialize weights to reasonable defaults """
        self.input_block.reset_weights()
        self.backbone.reset_weights()
        self.q_head.reset_weights()

    def forward(self, observations):
        """ Model forward pass """
        observations = self.input_block(observations)
        base_output = self.backbone(observations)
        q_values = self.q_head(base_output)
        return q_values

    def step(self, observations):
        """ Sample action from an action space for given state """
        q_values = self(observations)
        actions = self.q_head.sample(q_values)

        return {
            'actions': actions,
            'q': q_values
        }

    def evaluate(self, rollout: Rollout) -> Evaluator:
        """ Evaluate model on a rollout """
        return QModelEvaluator(self, rollout)


class QModelFactory(ModelFactory):
    """ Factory class for q-learning models """
    def __init__(self, input_block: ModelFactory, backbone: ModelFactory):
        self.input_block = input_block
        self.backbone = backbone

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        input_block = self.input_block.instantiate()
        backbone = self.backbone.instantiate(**extra_args)

        return QModel(input_block, backbone, extra_args['action_space'])


def create(backbone: ModelFactory, input_block: typing.Optional[ModelFactory]=None):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return QModelFactory(input_block=input_block, backbone=backbone)
