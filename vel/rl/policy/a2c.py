import gym
import torch
import torch.nn.functional as F

from vel.metric.base import AveragingNamedMetric
from vel.util.stats import explained_variance
from vel.api import ModelFactory, BatchInfo, BackboneNetwork

from vel.rl.api import RlPolicy, Rollout, Trajectories
from vel.rl.discount_bootstrap import discount_bootstrap_gae
from vel.rl.module.stochastic_action_head import StochasticActionHead
from vel.rl.module.value_head import ValueHead


class A2C(RlPolicy):
    """ Simplest policy gradient - calculate loss as an advantage of an actor versus value function """
    def __init__(self, net: BackboneNetwork, action_space: gym.Space,
                 entropy_coefficient, value_coefficient, discount_factor: float,
                 gae_lambda=1.0):
        super().__init__(discount_factor)

        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.gae_lambda = gae_lambda

        self.net = net

        assert not self.net.is_stateful, "For stateful policies, use A2CRnn"

        # Make sure network returns two results
        (action_size, value_size) = self.net.size_hints().assert_tuple(2)

        self.action_head = StochasticActionHead(
            action_space=action_space,
            input_dim=action_size.last(),
        )

        self.value_head = ValueHead(
            input_dim=value_size.last()
        )

    def reset_weights(self):
        """ Initialize properly model weights """
        self.net.reset_weights()
        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observation, state=None):
        """ Calculate model outputs """
        action_hidden, value_hidden = self.net(observation, state=state)
        return self.action_head(action_hidden), self.value_head(value_hidden)

    def act(self, observation, state=None, deterministic=False):
        """ Select actions based on model's output """
        action_pd_params, value_output = self(observation)
        actions = self.action_head.sample(action_pd_params, deterministic=deterministic)

        # log likelihood of selected action
        logprobs = self.action_head.logprob(actions, action_pd_params)

        return {
            'actions': actions,
            'values': value_output,
            'action:logprobs': logprobs
        }

    def process_rollout(self, rollout: Rollout) -> Rollout:
        """ Process rollout for optimization before any chunking/shuffling  """
        assert isinstance(rollout, Trajectories), "A2C requires trajectory rollouts"

        advantages = discount_bootstrap_gae(
            rewards_buffer=rollout.transition_tensors['rewards'],
            dones_buffer=rollout.transition_tensors['dones'],
            values_buffer=rollout.transition_tensors['values'],
            final_values=rollout.rollout_tensors['final_values'],
            discount_factor=self.discount_factor,
            gae_lambda=self.gae_lambda,
            number_of_steps=rollout.num_steps
        )

        returns = advantages + rollout.transition_tensors['values']

        rollout.transition_tensors['advantages'] = advantages
        rollout.transition_tensors['returns'] = returns

        return rollout

    def calculate_gradient(self, batch_info: BatchInfo, rollout: Rollout) -> dict:
        """ Calculate loss of the supplied rollout """
        observations = rollout.batch_tensor('observations')

        actions = rollout.batch_tensor('actions')
        advantages = rollout.batch_tensor('advantages')
        returns = rollout.batch_tensor('returns')
        rollout_values = rollout.batch_tensor('values')

        pd_params, model_values = self(observations)

        log_probs = self.action_head.logprob(actions, pd_params)
        entropy = self.action_head.entropy(pd_params)

        # Actual calculations. Pretty trivial
        policy_loss = -torch.mean(advantages * log_probs)
        value_loss = 0.5 * F.mse_loss(model_values, returns)
        policy_entropy = torch.mean(entropy)

        loss_value = (
            policy_loss - self.entropy_coefficient * policy_entropy + self.value_coefficient * value_loss
        )

        loss_value.backward()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'policy_entropy': policy_entropy.item(),
            'advantage_norm': torch.norm(advantages).item(),
            'explained_variance': explained_variance(returns, rollout_values)
        }

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("value_loss", scope="model"),
            AveragingNamedMetric("policy_entropy", scope="model"),
            AveragingNamedMetric("policy_loss", scope="model"),
            AveragingNamedMetric("advantage_norm", scope="model"),
            AveragingNamedMetric("explained_variance", scope="model")
        ]


class A2CFactory(ModelFactory):
    """ Factory class for policy gradient models """
    def __init__(self, net, entropy_coefficient, value_coefficient, discount_factor, gae_lambda=1.0):
        self.net = net
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        action_space = extra_args.pop('action_space')
        net = self.net.instantiate(**extra_args)

        return A2C(
            net=net,
            action_space=action_space,
            entropy_coefficient=self.entropy_coefficient,
            value_coefficient=self.value_coefficient,
            discount_factor=self.discount_factor,
            gae_lambda=self.gae_lambda
        )


def create(net: ModelFactory, entropy_coefficient, value_coefficient, discount_factor, gae_lambda=1.0):
    """ Vel factory function """
    return A2CFactory(
        net=net,
        entropy_coefficient=entropy_coefficient,
        value_coefficient=value_coefficient,
        discount_factor=discount_factor,
        gae_lambda=gae_lambda
    )
