import torch
import torch.nn.functional as F

from vel.api.metrics.averaging_metric import AveragingNamedMetric
from vel.rl.api.base import OptimizerAlgoBase


def select_indices(tensor, indices):
    """ Select indices from tensor """
    return tensor.gather(1, indices.unsqueeze(1)).squeeze()


class AcerPolicyGradient(OptimizerAlgoBase):
    """ Actor-Critic with Experience Replay - policy gradient calculations """

    def __init__(self, model_factory, trust_region: bool=True, entropy_coefficient: float=0.01,
                 q_coefficient: float=0.5, rho_cap: float=10.0, retrace_rho_cap: float=1.0, max_grad_norm: float=None,
                 average_model_alpha=0.99, trust_region_delta=1.0):
        super().__init__(max_grad_norm)

        self.discount_factor = None
        self.number_of_steps = None

        self.trust_region = trust_region
        self.model_factory = model_factory

        self.entropy_coefficient = entropy_coefficient
        self.q_coefficient = q_coefficient

        self.rho_cap = rho_cap
        self.retrace_rho_cap = retrace_rho_cap

        # Trust region settings
        self.average_model = None
        self.average_model_initialized = False
        self.average_model_alpha = average_model_alpha
        self.trust_region_delta = trust_region_delta

    def initialize(self, settings, model, environment, device):
        """ Initialize policy gradient from reinforcer settings """
        self.discount_factor = settings.discount_factor
        self.number_of_steps = settings.number_of_steps

        if self.trust_region:
            self.average_model = self.model_factory.instantiate(action_space=environment.action_space).to(device)

    def update_average_model(self, model):
        """ Update weights of the average model with new model observation """
        if not self.average_model_initialized:
            # Initialize average model to have the same weights as the main model
            self.average_model.load_state_dict(model.state_dict())
            self.average_model_initialized = True
        else:
            for model_param, average_param in zip(model.parameters(), self.average_model.parameters()):
                # EWMA average model update
                average_param.data.mul_(self.average_model_alpha).add_(model_param.data * (1 - self.average_model_alpha))

    def calculate_gradient(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        local_epsilon = 1e-6

        actions = rollout['actions']
        rewards = rollout['rewards']
        dones = rollout['dones']
        observations = rollout['observations']
        final_values = rollout['final_values']
        rollout_probabilities = torch.exp(rollout['action_logits'])

        # We calculate the trust-region update with respect to the average model
        if self.trust_region:
            self.update_average_model(model)

        action_logits, q_outputs = model(observations)
        q_selected = select_indices(q_outputs, actions)

        # We only want to propagate gradients through specific variables
        with torch.no_grad():
            # Initialize few
            model_probabilities = torch.exp(action_logits)

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

        neglogps = F.nll_loss(action_logits, actions, reduction='none')  # f_i
        policy_gradient_loss = torch.mean(advantages * importance_sampling_coefficient * neglogps)

        # Policy gradient bias correction
        with torch.no_grad():
            advantages_bias_correction = q_outputs - model_state_values.view(model_state_values.size(0), 1)
            bias_correction_coefficient = F.relu(1.0 - self.rho_cap / (rho + local_epsilon))

        # This sum is an expectation with respect to action probabilities according to model policy
        policy_gradient_bias_correction_gain = torch.sum(
            action_logits * bias_correction_coefficient * advantages_bias_correction * model_probabilities,
            dim=1
        )

        policy_gradient_bias_correction_loss = - torch.mean(policy_gradient_bias_correction_gain)

        policy_loss = policy_gradient_loss + policy_gradient_bias_correction_loss

        q_function_loss = 0.5 * F.mse_loss(q_selected, q_retraced)

        if self.trust_region:
            with torch.no_grad():
                average_action_logits, _ = self.average_model(observations)

            actor_loss = policy_loss - self.entropy_coefficient * policy_entropy
            q_loss = self.q_coefficient * q_function_loss

            actor_gradient = torch.autograd.grad(-actor_loss, action_logits, retain_graph=True)[0]

            # kl_divergence = model.kl_divergence(average_action_logits, action_logits).mean()
            # kl_divergence_grad = torch.autograd.grad(kl_divergence, action_logits, retain_graph=True)

            # Analytically calculated derivative of KL divergence on logits
            # That makes it hardcoded for discrete action spaces
            kl_divergence_grad_symbolic = - torch.exp(average_action_logits) / action_logits.size(0)

            k_dot_g = (actor_gradient * kl_divergence_grad_symbolic).sum(dim=-1)
            k_dot_k = (kl_divergence_grad_symbolic ** 2).sum(dim=-1)

            adjustment = (k_dot_g - self.trust_region_delta) / k_dot_k
            adjustment_clipped = adjustment.clamp(min=0.0)

            actor_gradient_updated = actor_gradient - adjustment_clipped.view(adjustment_clipped.size(0), 1)

            # Populate gradient from the newly updated fn
            action_logits.backward(gradient=-actor_gradient_updated, retain_graph=True)
            q_loss.backward(retain_graph=True)
        else:
            # Just populate gradient from the loss
            loss = policy_loss + self.q_coefficient * q_function_loss - self.entropy_coefficient * policy_entropy

            loss.backward()

        return {
            'policy_loss': policy_loss.item(),
            'policy_gradient_loss': policy_gradient_loss.item(),
            'policy_gradient_bias_correction': policy_gradient_bias_correction_loss.item(),
            'avg_q_selected': q_selected.mean().item(),
            'avg_q_retraced': q_retraced.mean().item(),
            'q_loss': q_function_loss.item(),
            'policy_entropy': policy_entropy.item(),
            'advantage_norm': torch.norm(advantages).item(),
            'explained_variance': explained_variance.item(),
            'model_prob_std': model_probabilities.std().item(),
            'rollout_prob_std': rollout_probabilities.std().item()
        }

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
            AveragingNamedMetric("advantage_norm"),
            AveragingNamedMetric("grad_norm"),
            AveragingNamedMetric("model_prob_std"),
            AveragingNamedMetric("rollout_prob_std"),
            AveragingNamedMetric("avg_q_selected"),
            AveragingNamedMetric("avg_q_retraced")
        ]


def create(model, trust_region, entropy_coefficient, q_coefficient, max_grad_norm, rho_cap=10.0, retrace_rho_cap=1.0,
           average_model_alpha=0.99, trust_region_delta=1.0):
    return AcerPolicyGradient(
        trust_region=trust_region,
        model_factory=model,
        entropy_coefficient=entropy_coefficient,
        q_coefficient=q_coefficient,
        rho_cap=rho_cap,
        retrace_rho_cap=retrace_rho_cap,
        max_grad_norm=max_grad_norm,
        average_model_alpha=average_model_alpha,
        trust_region_delta=trust_region_delta
    )
