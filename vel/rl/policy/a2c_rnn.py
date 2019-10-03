import gym
import torch
import torch.nn.functional as F

from vel.api import ModelFactory, BatchInfo, BackboneNetwork
from vel.metric.base import AveragingNamedMetric
from vel.rl.api import RlPolicy, Rollout, Trajectories
from vel.rl.discount_bootstrap import discount_bootstrap_gae
from vel.rl.module.stochastic_rnn_policy import StochasticRnnPolicy
from vel.util.situational import gym_space_to_size_hint
from vel.util.stats import explained_variance


class A2CRnn(RlPolicy):
    """ Simplest policy gradient - calculate loss as an advantage of an actor versus value function """
    def __init__(self, net: BackboneNetwork, action_space: gym.Space,
                 entropy_coefficient, value_coefficient, discount_factor: float,
                 gae_lambda=1.0):
        super().__init__(discount_factor)

        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.gae_lambda = gae_lambda

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
        assert isinstance(rollout, Trajectories), "A2C requires trajectory rollouts"

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

        # rollout values
        actions = rollout.batch_tensor('actions')
        advantages = rollout.batch_tensor('advantages')
        returns = rollout.batch_tensor('returns')
        rollout_values = rollout.batch_tensor('values')

        # Let's evaluate the model
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

        log_probs = self.net.action_head.logprob(actions, pd_params)
        entropy = self.net.action_head.entropy(pd_params)

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


class A2CRnnFactory(ModelFactory):
    """ Factory class for policy gradient models """
    def __init__(self, net_factory, entropy_coefficient, value_coefficient, discount_factor, gae_lambda=1.0):
        self.net_factory = net_factory
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        action_space = extra_args.pop('action_space')
        observation_space = extra_args.pop('observation_space')

        size_hint = gym_space_to_size_hint(observation_space)

        net = self.net_factory.instantiate(size_hint=size_hint, **extra_args)

        return A2CRnn(
            net=net,
            action_space=action_space,
            entropy_coefficient=self.entropy_coefficient,
            value_coefficient=self.value_coefficient,
            discount_factor=self.discount_factor,
            gae_lambda=self.gae_lambda
        )


def create(net: ModelFactory, entropy_coefficient, value_coefficient, discount_factor, gae_lambda=1.0):
    """ Vel factory function """
    return A2CRnnFactory(
        net_factory=net,
        entropy_coefficient=entropy_coefficient,
        value_coefficient=value_coefficient,
        discount_factor=discount_factor,
        gae_lambda=gae_lambda
    )
