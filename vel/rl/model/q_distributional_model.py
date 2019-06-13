import gym
import typing

from vel.api import LinearBackboneModel, ModelFactory, BackboneModel
from vel.modules.input.identity import IdentityFactory
from vel.rl.api import Rollout, RlModel, Evaluator
from vel.rl.modules.q_distributional_head import QDistributionalHead


class QDistributionalModelEvaluator(Evaluator):
    """ Evaluate distributional q-model """
    def __init__(self, model: 'QDistributionalModel', rollout: Rollout):
        super().__init__(rollout)
        self.model = model

    @Evaluator.provides('model:q')
    def model_q(self):
        """ Action values for all (discrete) actions """
        # observations = self.get('rollout:observations')
        # # This mean of last dimension collapses the histogram/calculates mean reward
        # return self.model(observations).mean(dim=-1)
        raise NotImplementedError

    @Evaluator.provides('model:q_dist')
    def model_q_dist(self):
        """ Action values for all (discrete) actions """
        observations = self.get('rollout:observations')
        # This mean of last dimension collapses the histogram/calculates mean reward
        return self.model(observations)

    @Evaluator.provides('model:action:q')
    def model_action_q(self):
        """ Action values for selected actions in the rollout """
        raise NotImplementedError

    @Evaluator.provides('model:action:q_dist')
    def model_action_q_dist(self):
        """ Action values for selected actions in the rollout """
        q = self.get('model:q_dist')
        actions = self.get('rollout:actions')
        return q[range(q.size(0)), actions]

    @Evaluator.provides('model:q_next')
    def model_q_next(self):
        """ Action values for all (discrete) actions """
        raise NotImplementedError

    @Evaluator.provides('model:q_dist_next')
    def model_q_dist_next(self):
        """ Action values for all (discrete) actions """
        observations = self.get('rollout:observations_next')
        # This mean of last dimension collapses the histogram/calculates mean reward
        return self.model(observations)


class QDistributionalModel(RlModel):
    """
    A deterministic greedy action-value model that learns a value function distribution rather than
    just an expectation.
    Supports only discrete action spaces (ones that can be enumerated)
    """
    def __init__(self, input_block: BackboneModel, backbone: LinearBackboneModel, action_space: gym.Space,
                 vmin: float, vmax: float, atoms: int=1):
        super().__init__()

        self.action_space = action_space

        self.input_block = input_block
        self.backbone = backbone

        self.q_head = QDistributionalHead(
            input_dim=backbone.output_dim, action_space=action_space,
            vmin=vmin, vmax=vmax,
            atoms=atoms
        )

    def reset_weights(self):
        """ Initialize weights to reasonable defaults """
        self.input_block.reset_weights()
        self.backbone.reset_weights()
        self.q_head.reset_weights()

    def forward(self, observations):
        """ Model forward pass """
        input_data = self.input_block(observations)
        base_output = self.backbone(input_data)
        log_histogram = self.q_head(base_output)
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
    def __init__(self, input_block: ModelFactory, backbone: ModelFactory, vmin: float, vmax: float, atoms: int):
        self.input_block = input_block
        self.backbone = backbone
        self.vmin = vmin
        self.vmax = vmax
        self.atoms = atoms

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        input_block = self.input_block.instantiate()
        backbone = self.backbone.instantiate(**extra_args)

        return QDistributionalModel(
            input_block=input_block,
            backbone=backbone,
            action_space=extra_args['action_space'],
            vmin=self.vmin,
            vmax=self.vmax,
            atoms=self.atoms
        )


def create(backbone: ModelFactory, vmin: float, vmax: float, atoms: int,
           input_block: typing.Optional[ModelFactory]=None):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return QDistributionalModelFactory(
        input_block=input_block, backbone=backbone,
        vmin=vmin,
        vmax=vmax,
        atoms=atoms
    )
