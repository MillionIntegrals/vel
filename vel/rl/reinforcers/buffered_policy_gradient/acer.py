import torch
import torch.nn.functional as F

from vel.api.metrics.averaging_metric import AveragingNamedMetric
from vel.rl.reinforcers.policy_gradient.policy_gradient_reinforcer import PolicyGradientBase


def select_indices(tensor, indices):
    """ Select indices from tensor """
    return tensor.gather(1, indices.unsqueeze(1)).squeeze()


class AcerPolicyGradient(PolicyGradientBase):
    """ Actor-Critic with Experience Replay - policy gradient calculations """

    def __init__(self, entropy_coefficient: float=0.01, q_coefficient: float=0.5,
                 rho_cap: float=10.0, retrace_rho_cap: float=1.0):
        super().__init__()

        self.discount_factor = None
        self.number_of_steps = None

        self.entropy_coefficient = entropy_coefficient
        self.q_coefficient = q_coefficient

        self.rho_cap = rho_cap
        self.retrace_rho_cap = retrace_rho_cap

    def initialize(self, settings):
        """ Initialize policy gradient from reinforcer settings """
        self.discount_factor = settings.discount_factor
        self.number_of_steps = settings.number_of_steps

    def calculate_loss(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        local_epsilon = 1e-6

        actions = rollout['actions']
        rewards = rollout['rewards']
        dones = rollout['dones']
        observations = rollout['observations']
        final_values = rollout['final_values']

        action_logits, q_outputs = model(observations)
        q_selected = select_indices(q_outputs, actions)

        # We only want to propagate gradients through specific variables
        with torch.no_grad():
            # Initialize few
            model_probabilities = torch.exp(action_logits)
            rollout_probabilities = torch.exp(rollout['action_logits'])

            # Importance sampling correction - we must find the quotient of probabilities
            rho = model_probabilities / (rollout_probabilities + local_epsilon)

            # Probability quotient only for selected actions
            rho_selected = select_indices(rho, actions)

            # Q values for selected actions
            model_state_values = (model_probabilities * q_outputs).sum(dim=1)
            q_retraced = self.retrace(rewards, dones, q_selected, model_state_values, rho_selected, final_values)

            advantages = q_retraced - model_state_values
            importance_sampling_coefficient = torch.min(rho_selected, self.rho_cap * torch.ones_like(rho_selected))

            explained_variance = 1 - torch.var(q_retraced - q_selected) / torch.var(q_retraced)

        # Entropy of the policy distribution
        policy_entropy = torch.mean(model.entropy(action_logits))

        neglogps = F.nll_loss(action_logits, actions, reduction='none')
        policy_gradient_loss = torch.mean(advantages * importance_sampling_coefficient * neglogps)

        # Policy gradient bias correction
        with torch.no_grad():
            advantages_bias_correction = q_outputs - model_state_values.view(model_state_values.size(0), 1)
            bias_correction_coefficient = F.relu(1.0 - self.rho_cap / (rho + local_epsilon))

        # This sum is an expectation with respect to action probabilities according to model policy
        policy_gradient_bias_correction_gain = torch.sum(action_logits * bias_correction_coefficient * advantages_bias_correction * model_probabilities, dim=1)

        policy_gradient_bias_correction = - torch.mean(policy_gradient_bias_correction_gain)

        policy_loss = policy_gradient_loss + policy_gradient_bias_correction

        q_function_loss = 0.5 * F.mse_loss(q_selected, q_retraced)

        loss = policy_loss + self.q_coefficient * q_function_loss - self.entropy_coefficient * policy_entropy

        batch_info['policy_gradient_data'].append({
            'policy_loss': policy_loss,
            'policy_gradient_loss': policy_gradient_loss,
            'policy_gradient_bias_correction': policy_gradient_bias_correction,
            'q_loss': q_function_loss,
            'policy_entropy': policy_entropy,
            'explained_variance': explained_variance
        })

        return loss

    def retrace(self, rewards, dones, q_values, state_values, rho, final_values):
        """ Calculate Q retraced targets """
        parallel_envs = rewards.size(0) // self.number_of_steps

        rewards = self._reshape_to_episodes(rewards, parallel_envs)
        dones = self._reshape_to_episodes(dones, parallel_envs).to(torch.float32)

        q_values = self._reshape_to_episodes(q_values, parallel_envs)
        state_values = self._reshape_to_episodes(state_values, parallel_envs)
        rho = self._reshape_to_episodes(rho, parallel_envs)

        rho_bar = torch.min(torch.ones_like(rho) * self.retrace_rho_cap, rho)

        q_retraced_buffer = torch.zeros_like(rewards)

        next_value = final_values

        for i in reversed(range(self.number_of_steps)):
            q_retraced = rewards[i] + self.discount_factor * next_value * (1.0 - dones[i])

            # Next iteration
            next_value = rho_bar[i] * (q_retraced - q_values[i]) + state_values[i]

            q_retraced_buffer[i] = q_retraced

        return q_retraced_buffer.flatten()

    def _reshape_to_episodes(self, array, parallel_envs):
        new_shape = tuple([self.number_of_steps, parallel_envs] + list(array.shape[2:]))
        return array.view(new_shape)

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("q_loss"),
            AveragingNamedMetric("policy_entropy"),
            AveragingNamedMetric("policy_loss"),
            AveragingNamedMetric("policy_gradient_loss"),
            AveragingNamedMetric("policy_gradient_bias_correction"),
            AveragingNamedMetric("explained_variance"),
        ]


def create(entropy_coefficient, q_coefficient, rho_cap=10.0, retrace_rho_cap=1.0):
    return AcerPolicyGradient(
        entropy_coefficient=entropy_coefficient,
        q_coefficient=q_coefficient,
        rho_cap=rho_cap,
        retrace_rho_cap=retrace_rho_cap
    )
