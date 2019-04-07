import gym
import typing

from vel.api import LinearBackboneModel, Model, ModelFactory, BackboneModel
from vel.modules.input.identity import IdentityFactory
from vel.rl.api import Rollout, Evaluator
from vel.rl.models.q_distributional_model import QDistributionalModelEvaluator
from vel.rl.modules.q_distributional_noisy_dueling_head import QDistributionalNoisyDuelingHead


class QRainbowModel(Model):
    """
    A deterministic greedy action-value model.
    Includes following commonly known modifications:
    - Distributional Q-Learning
    - Dueling architecture
    - Noisy Nets
    """

    def __init__(self, input_block: BackboneModel, backbone: LinearBackboneModel, action_space: gym.Space, vmin: float,
                 vmax: float, atoms: int = 1, initial_std_dev: float = 0.4, factorized_noise: bool = True):
        super().__init__()

        self.action_space = action_space

        self.input_block = input_block
        self.backbone = backbone

        self.q_head = QDistributionalNoisyDuelingHead(
            input_dim=backbone.output_dim,
            action_space=action_space,
            vmin=vmin, vmax=vmax, atoms=atoms,
            initial_std_dev=initial_std_dev, factorized_noise=factorized_noise
        )

    def reset_weights(self):
        """ Initialize weights to reasonable defaults """
        self.input_block.reset_weights()
        self.backbone.reset_weights()
        self.q_head.reset_weights()

    def forward(self, observations):
        """ Model forward pass """
        input_data = self.input_block(observations)
        advantage_features, value_features = self.backbone(input_data)
        log_histogram = self.q_head(advantage_features, value_features)
        return log_histogram

    def histogram_info(self):
        """ Return extra information about histogram """
        return self.q_head.histogram_info()

    def step(self, observations):
        """ Sample action from an action space for given state """
        log_histogram = self(observations)
        actions = self.q_head.sample(log_histogram)

        return {
            'actions': actions,
            'log_histogram': log_histogram
        }

    def evaluate(self, rollout: Rollout) -> Evaluator:
        """ Evaluate model on a rollout """
        return QDistributionalModelEvaluator(self, rollout)


class QDistributionalModelFactory(ModelFactory):
    """ Factory class for q-learning models """
    def __init__(self, input_block: ModelFactory, backbone: ModelFactory, vmin: float, vmax: float, atoms: int,
                 initial_std_dev: float = 0.4, factorized_noise: bool = True):
        self.input_block = input_block
        self.backbone = backbone
        self.vmin = vmin
        self.vmax = vmax
        self.atoms = atoms
        self.initial_std_dev = initial_std_dev
        self.factorized_noise = factorized_noise

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        input_block = self.input_block.instantiate()
        backbone = self.backbone.instantiate(**extra_args)

        return QRainbowModel(
            input_block=input_block,
            backbone=backbone,
            action_space=extra_args['action_space'],
            vmin=self.vmin,
            vmax=self.vmax,
            atoms=self.atoms,
            initial_std_dev=self.initial_std_dev,
            factorized_noise=self.factorized_noise
        )


def create(backbone: ModelFactory, vmin: float, vmax: float, atoms: int, initial_std_dev: float = 0.4,
           factorized_noise: bool = True, input_block: typing.Optional[ModelFactory] = None):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return QDistributionalModelFactory(
        input_block=input_block, backbone=backbone,
        vmin=vmin,
        vmax=vmax,
        atoms=atoms,
        initial_std_dev=initial_std_dev,
        factorized_noise=factorized_noise
    )
