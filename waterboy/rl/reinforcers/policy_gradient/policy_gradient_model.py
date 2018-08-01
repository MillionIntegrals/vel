import numpy as np

from waterboy.api.base import LinearBackboneModel, Model, ModelAugmentor
from waterboy.exceptions import WaterboyException
from waterboy.openai.baselines.common.vec_env import VecEnv
from waterboy.rl.modules.action_head import ActionHead
from waterboy.rl.modules.value_head import ValueHead


class PolicyGradientModel(Model):
    """ For a policy gradient algorithm we need set of custom heads for our model """

    def __init__(self, base_model: LinearBackboneModel, environment: VecEnv):
        super().__init__()

        self.base_model = base_model
        self.action_head = ActionHead(action_space=environment.action_space, input_dim=self.base_model.output_dim)
        self.value_head = ValueHead(self.base_model.output_dim)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.base_model.reset_weights()
        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observations):
        base_output = self.base_model(observations)

        action_output = self.action_head(base_output)
        value_output = self.value_head(base_output)

        return action_output, value_output

    def loss_value(self, x_data, y_true, y_pred):
        raise WaterboyException("Invalid method to call for this model")

    def step(self, observation):
        """ Select actions based on model's output """
        action_logits, value_output = self(observation)
        actions = self.action_head.sample(action_logits)

        local_actions = actions.detach().cpu().numpy()

        # - log probability
        neglogp = -action_logits[np.vstack([np.arange(actions.size(0)), local_actions])]

        return actions, value_output, neglogp

    def value(self, observation):
        base_output = self.base_model(observation)
        value_output = self.value_head(base_output)
        return value_output

    def entropy(self, action_logits):
        return self.action_head.entropy(action_logits)


class PolicyGradientModelAugmentor(ModelAugmentor):
    """ Factory  class for policy gradient models """

    def augment(self, base_model: Model, extra_info: dict=None) -> Model:
        """ Create new policy gradient model"""
        return PolicyGradientModel(base_model, extra_info['env'])


def create():
    return PolicyGradientModelAugmentor()
