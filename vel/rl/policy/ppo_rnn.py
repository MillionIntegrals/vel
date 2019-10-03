import numbers

import gym
import torch

from vel.api import BatchInfo, ModelFactory, BackboneNetwork
from vel.function.constant import ConstantSchedule
from vel.metric.base import AveragingNamedMetric
from vel.rl.api import RlPolicy, Rollout, Trajectories
from vel.rl.discount_bootstrap import discount_bootstrap_gae
from vel.rl.module.stochastic_rnn_policy import StochasticRnnPolicy
from vel.util.situational import gym_space_to_size_hint
from vel.util.stats import explained_variance


class PPORnn(RlPolicy):
    """ Proximal Policy Optimization - https://arxiv.org/abs/1707.06347 """
    def __init__(self, net: BackboneNetwork, action_space: gym.Space,
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

        self.net = StochasticRnnPolicy(net, action_space)

        assert self.net.is_stateful, "Policy must be stateful"

    def reset_weights(self):
        """ Initialize properly model weights """
        self.net.reset_weights()

    def forward(self, observation, state=None):
        """ Calculate model outputs """
        return self.net(observation, state=state)

    def is_stateful(self) -> bool:
        return self.net.is_stateful

    def zero_state(self, batch_size):
        return self.net.zero_state(batch_size)

    def reset_state(self, state, dones):
        return self.net.reset_state(state, dones)

    def act(self, observation, state=None, deterministic=False):
        """ Select actions based on model's output """
        action_pd_params, value_output, next_state = self(observation, state=state)
        actions = self.net.action_head.sample(action_pd_params, deterministic=deterministic)

        # log likelihood of selected action
        logprobs = self.net.action_head.logprob(actions, action_pd_params)

        return {
            'action:logprobs': logprobs,
            'actions': actions,
            'state': next_state,
            'values': value_output,
        }

    def process_rollout(self, rollout: Rollout) -> Rollout:
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

    def _extract_initial_state(self, transition_tensors):
        """ Extract initial state from the state dictionary """
        state = {}

        idx = len('state') + 1

        for key, value in transition_tensors.items():
            if key.startswith('state'):
                state[key[idx:]] = value[0]

        return state

    def calculate_gradient(self, batch_info: BatchInfo, rollout: Rollout) -> dict:
        """ Calculate loss of the supplied rollout """
        assert isinstance(rollout, Trajectories), "For an RNN model, we must evaluate trajectories"

        # Part 0.0 - Rollout values
        actions = rollout.batch_tensor('actions')
        advantages = rollout.batch_tensor('advantages')
        returns = rollout.batch_tensor('returns')
        rollout_values = rollout.batch_tensor('values')
        rollout_action_logprobs = rollout.batch_tensor('action:logprobs')

        # PART 0.1 - Model evaluation
        observations = rollout.transition_tensors['observations']
        hidden_state = self._extract_initial_state(rollout.transition_tensors)
        dones = rollout.transition_tensors['dones']

        action_accumulator = []
        value_accumulator = []

        # Evaluate recurrent network step by step
        for i in range(observations.size(0)):
            action_output, value_output, hidden_state = self(observations[i], hidden_state)
            hidden_state = self.reset_state(hidden_state, dones[i])

            action_accumulator.append(action_output)
            value_accumulator.append(value_output)

        pd_params = torch.cat(action_accumulator, dim=0)
        model_values = torch.cat(value_accumulator, dim=0)

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


class PPORnnFactory(ModelFactory):
    """ Factory class for policy gradient models """
    def __init__(self, net_factory,
                 entropy_coefficient, value_coefficient, cliprange, discount_factor: float,
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

        return PPORnn(
            net=net,
            entropy_coefficient=self.entropy_coefficient,
            action_space=action_space,
            value_coefficient=self.value_coefficient,
            cliprange=self.cliprange,
            discount_factor=self.discount_factor,
            normalize_advantage=self.normalize_advantage,
            gae_lambda=self.gae_lambda,
        )


def create(net: ModelFactory,
           entropy_coefficient, value_coefficient, cliprange, discount_factor: float,
           normalize_advantage: bool = True, gae_lambda: float = 1.0):
    """ Vel factory function """
    return PPORnnFactory(
        net_factory=net,
        entropy_coefficient=entropy_coefficient,
        value_coefficient=value_coefficient,
        cliprange=cliprange,
        discount_factor=discount_factor,
        normalize_advantage=normalize_advantage,
        gae_lambda=gae_lambda
    )

