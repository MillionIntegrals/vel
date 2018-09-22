import gym

from vel.api.base import LinearBackboneModel, Model, ModelFactory
from vel.rl.modules.action_head import ActionHead
from vel.rl.modules.value_head import ValueHead


class PolicyGradientModel(Model):
    """ For a policy gradient algorithm we need set of custom heads for our model """

    def __init__(self, backbone: LinearBackboneModel, action_space: gym.Space):
        super().__init__()

        self.backbone = backbone
        self.action_head = ActionHead(
            action_space=action_space,
            input_dim=self.backbone.output_dim
        )
        self.value_head = ValueHead(input_dim=self.backbone.output_dim)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.backbone.reset_weights()
        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observations):
        """ Calculate model outputs """
        base_output = self.backbone(observations)

        action_output = self.action_head(base_output)
        value_output = self.value_head(base_output)

        return action_output, value_output

    def step(self, observation, argmax_sampling=False):
        """ Select actions based on model's output """
        action_pd_params, value_output = self(observation)
        actions = self.action_head.sample(action_pd_params, argmax_sampling=argmax_sampling)

        # log likelihood of selected action
        logprob = self.action_head.logprob(actions, action_pd_params)

        return {
            'actions': actions,
            'values': value_output,
            'logprob': logprob
        }

    def logprob(self, action_sample, action_params):
        """ Calculate - log(prob) of selected actions """
        return self.action_head.logprob(action_sample, action_params)

    def value(self, observation):
        """ Calculate only value head for given state """
        base_output = self.backbone(observation)
        value_output = self.value_head(base_output)
        return value_output

    def entropy(self, action_pd_params):
        """ Entropy of a probability distribution """
        return self.action_head.entropy(action_pd_params)


class PolicyGradientModelFactory(ModelFactory):
    """ Factory  class for policy gradient models """
    def __init__(self, backbone: ModelFactory):
        self.backbone = backbone

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        backbone = self.backbone.instantiate(**extra_args)
        return PolicyGradientModel(backbone, extra_args['action_space'])


def create(backbone: ModelFactory):
    """ Vel creation function """
    return PolicyGradientModelFactory(backbone=backbone)
