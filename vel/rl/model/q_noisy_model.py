import gym
import typing

from vel.api import LinearBackboneModel, ModelFactory, BackboneModel
from vel.modules.input.identity import IdentityFactory
from vel.rl.api import Rollout, RlModel, Evaluator
from vel.rl.models.q_model import QModelEvaluator
from vel.rl.modules.q_noisy_head import QNoisyHead


class NoisyQModel(RlModel):
    """
    NoisyNets action-value model.
    Supports only discrete action spaces (ones that can be enumerated)
    """

    def __init__(self, input_block: BackboneModel, backbone: LinearBackboneModel, action_space: gym.Space,
                 initial_std_dev=0.4, factorized_noise=True):
        super().__init__()

        self.action_space = action_space

        self.input_block = input_block
        self.backbone = backbone
        self.q_head = QNoisyHead(
            input_dim=backbone.output_dim, action_space=action_space, initial_std_dev=initial_std_dev,
            factorized_noise=factorized_noise
        )

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


class NoisyQModelFactory(ModelFactory):
    """ Factory class for q-learning models """
    def __init__(self, input_block: ModelFactory, backbone: ModelFactory, initial_std_dev=0.4, factorized_noise=True):
        self.initial_std_dev = initial_std_dev
        self.factorized_noise = factorized_noise

        self.input_block = input_block
        self.backbone = backbone

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        input_block = self.input_block.instantiate()
        backbone = self.backbone.instantiate(**extra_args)

        return NoisyQModel(
            input_block, backbone, extra_args['action_space'], initial_std_dev=self.initial_std_dev,
            factorized_noise=self.factorized_noise
        )


def create(backbone: ModelFactory, input_block: typing.Optional[ModelFactory]=None, initial_std_dev=0.4,
           factorized_noise=True):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return NoisyQModelFactory(
        input_block=input_block, backbone=backbone, initial_std_dev=initial_std_dev, factorized_noise=factorized_noise
    )
