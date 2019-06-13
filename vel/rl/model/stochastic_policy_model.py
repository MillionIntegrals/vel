import gym
import typing

from vel.api import LinearBackboneModel, ModelFactory, BackboneModel
from vel.modules.input.identity import IdentityFactory
from vel.rl.api import Rollout, Evaluator, RlModel
from vel.rl.modules.action_head import ActionHead
from vel.rl.modules.value_head import ValueHead


class StochasticPolicyEvaluator(Evaluator):
    """ Evaluator for a policy gradient model """

    def __init__(self, model: 'StochasticPolicyModel', rollout: Rollout):
        super().__init__(rollout)

        self.model = model

        policy_params, estimated_values = model(self.rollout.batch_tensor('observations'))

        self.provide('model:policy_params',  policy_params)
        self.provide('model:values', estimated_values)

    @Evaluator.provides('model:action:logprobs')
    def model_action_logprobs(self):
        actions = self.get('rollout:actions')
        policy_params = self.get('model:policy_params')
        return self.model.action_head.logprob(actions, policy_params)

    @Evaluator.provides('model:entropy')
    def model_entropy(self):
        policy_params = self.get('model:policy_params')
        return self.model.entropy(policy_params)


class StochasticPolicyModel(RlModel):
    """
    Most generic policy gradient model class with a set of common actor-critic heads that share a single backbone
    """

    def __init__(self, input_block: BackboneModel, backbone: LinearBackboneModel, action_space: gym.Space):
        super().__init__()

        self.input_block = input_block
        self.backbone = backbone
        self.action_head = ActionHead(
            action_space=action_space,
            input_dim=self.backbone.output_dim
        )
        self.value_head = ValueHead(input_dim=self.backbone.output_dim)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.input_block.reset_weights()
        self.backbone.reset_weights()
        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observations):
        """ Calculate model outputs """
        input_data = self.input_block(observations)

        base_output = self.backbone(input_data)

        action_output = self.action_head(base_output)
        value_output = self.value_head(base_output)

        return action_output, value_output

    def step(self, observation, argmax_sampling=False):
        """ Select actions based on model's output """
        action_pd_params, value_output = self(observation)
        actions = self.action_head.sample(action_pd_params, argmax_sampling=argmax_sampling)

        # log likelihood of selected action
        logprobs = self.action_head.logprob(actions, action_pd_params)

        return {
            'actions': actions,
            'values': value_output,
            'action:logprobs': logprobs
        }

    def evaluate(self, rollout: Rollout) -> Evaluator:
        """ Evaluate model on a rollout """
        return StochasticPolicyEvaluator(self, rollout)

    def logprob(self, action_sample, policy_params):
        """ Calculate - log(prob) of selected actions """
        return self.action_head.logprob(action_sample, policy_params)

    def value(self, observations):
        """ Calculate only value head for given state """
        input_data = self.input_block(observations)
        base_output = self.backbone(input_data)
        value_output = self.value_head(base_output)
        return value_output

    def entropy(self, policy_params):
        """ Entropy of a probability distribution """
        return self.action_head.entropy(policy_params)


class StochasticPolicyModelFactory(ModelFactory):
    """ Factory class for policy gradient models """
    def __init__(self, input_block: IdentityFactory, backbone: ModelFactory):
        self.backbone = backbone
        self.input_block = input_block

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        input_block = self.input_block.instantiate()
        backbone = self.backbone.instantiate(**extra_args)

        return StochasticPolicyModel(input_block, backbone, extra_args['action_space'])


def create(backbone: ModelFactory, input_block: typing.Optional[ModelFactory]=None):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return StochasticPolicyModelFactory(input_block=input_block, backbone=backbone)
