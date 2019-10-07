import gym
import torch

import numbers

from vel.api import BatchInfo, ModelFactory, BackboneModule
from vel.util.situational import gym_space_to_size_hint
from vel.util.stats import explained_variance
from vel.function.constant import ConstantSchedule
from vel.metric.base import AveragingNamedMetric

from vel.rl.api import RlPolicy, Rollout, Trajectories
from vel.rl.discount_bootstrap import discount_bootstrap_gae
from vel.rl.module.stochastic_policy import StochasticPolicy


class PPO(RlPolicy):
    """ Proximal Policy Optimization - https://arxiv.org/abs/1707.06347 """
    def __init__(self, net: BackboneModule, action_space: gym.Space,
                 entropy_coefficient, value_coefficient, cliprange, discount_factor: float,
                 normalize_advantage: bool = True, gae_lambda: float = 1.0):
        super().__init__(discount_factor)

        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.normalize_advantage = normalize_advantage
        self.gae_lambda = gae_lambda

        if isinstance(cliprange, numbers.Number):
            self.cliprange = ConstantSchedule(cliprange)
        else:
            self.cliprange = cliprange

        self.net = StochasticPolicy(net, action_space)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.net.reset_weights()

    def forward(self, observation, state=None):
        """ Calculate model outputs """
        return self.net(observation)

    def act(self, observation, state=None, deterministic=False):
        """ Select actions based on model's output """
        action_pd_params, value_output = self(observation)
        actions = self.net.action_head.sample(action_pd_params, deterministic=deterministic)

        # log likelihood of selected action
        logprobs = self.net.action_head.logprob(actions, action_pd_params)

        return {
            'actions': actions,
            'values': value_output,
            'action:logprobs': logprobs
        }

    def process_rollout(self, rollout: Rollout):
        """ Process rollout for optimization before any chunking/shuffling  """
        assert isinstance(rollout, Trajectories), "PPO requires trajectory rollouts"

        advantages = discount_bootstrap_gae(
            rewards_buffer=rollout.transition_tensors['rewards'],
            dones_buffer=rollout.transition_tensors['dones'],
            values_buffer=rollout.transition_tensors['values'],
            final_values=rollout.rollout_tensors['final.values'],
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

        # Part 0.0 - Rollout values
        actions = rollout.batch_tensor('actions')
        advantages = rollout.batch_tensor('advantages')
        returns = rollout.batch_tensor('returns')
        rollout_values = rollout.batch_tensor('values')

        rollout_action_logprobs = rollout.batch_tensor('action:logprobs')

        # PART 0.1 - Model evaluation
        pd_params, model_values = self(observations)

        model_action_logprobs = self.net.action_head.logprob(actions, pd_params)
        entropy = self.net.action_head.entropy(pd_params)

        # Select the cliprange
        current_cliprange = self.cliprange.value(batch_info['progress'])

        # Normalize the advantages?
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PART 1 - policy entropy
        policy_entropy = torch.mean(entropy)

        # PART 2 - value function
        value_output_clipped = rollout_values + torch.clamp(
            model_values - rollout_values, -current_cliprange, current_cliprange
        )
        value_loss_part1 = (model_values - returns).pow(2)
        value_loss_part2 = (value_output_clipped - returns).pow(2)
        value_loss = 0.5 * torch.mean(torch.max(value_loss_part1, value_loss_part2))

        # PART 3 - policy gradient loss
        ratio = torch.exp(model_action_logprobs - rollout_action_logprobs)

        pg_loss_part1 = -advantages * ratio
        pg_loss_part2 = -advantages * torch.clamp(ratio, 1.0 - current_cliprange, 1.0 + current_cliprange)
        policy_loss = torch.mean(torch.max(pg_loss_part1, pg_loss_part2))

        loss_value = (
            policy_loss - self.entropy_coefficient * policy_entropy + self.value_coefficient * value_loss
        )

        loss_value.backward()

        with torch.no_grad():
            approx_kl_divergence = 0.5 * torch.mean((model_action_logprobs - rollout_action_logprobs).pow(2))
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > current_cliprange).to(dtype=torch.float))

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'policy_entropy': policy_entropy.item(),
            'approx_kl_divergence': approx_kl_divergence.item(),
            'clip_fraction': clip_fraction.item(),
            'advantage_norm': torch.norm(advantages).item(),
            'explained_variance': explained_variance(returns, rollout_values)
        }

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("policy_loss", scope="model"),
            AveragingNamedMetric("value_loss", scope="model"),
            AveragingNamedMetric("policy_entropy", scope="model"),
            AveragingNamedMetric("approx_kl_divergence", scope="model"),
            AveragingNamedMetric("clip_fraction", scope="model"),
            AveragingNamedMetric("advantage_norm", scope="model"),
            AveragingNamedMetric("explained_variance", scope="model")
        ]


class PPOFactory(ModelFactory):
    """ Factory class for policy gradient models """
    def __init__(self, net_factory, entropy_coefficient, value_coefficient, cliprange, discount_factor: float,
                 normalize_advantage: bool = True, gae_lambda: float = 1.0):
        self.net_factory = net_factory
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.cliprange = cliprange
        self.discount_factor = discount_factor
        self.normalize_advantage = normalize_advantage
        self.gae_lambda = gae_lambda

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        action_space = extra_args.pop('action_space')
        observation_space = extra_args.pop('observation_space')

        size_hint = gym_space_to_size_hint(observation_space)

        net = self.net_factory.instantiate(size_hint=size_hint, **extra_args)

        return PPO(
            net=net,
            action_space=action_space,
            entropy_coefficient=self.entropy_coefficient,
            value_coefficient=self.value_coefficient,
            cliprange=self.cliprange,
            discount_factor=self.discount_factor,
            normalize_advantage=self.normalize_advantage,
            gae_lambda=self.gae_lambda,
        )


def create(net: ModelFactory, entropy_coefficient, value_coefficient, cliprange, discount_factor: float,
           normalize_advantage: bool = True, gae_lambda: float = 1.0):
    """ Vel factory function """
    return PPOFactory(
        net_factory=net,
        entropy_coefficient=entropy_coefficient,
        value_coefficient=value_coefficient,
        cliprange=cliprange,
        discount_factor=discount_factor,
        normalize_advantage=normalize_advantage,
        gae_lambda=gae_lambda
    )

