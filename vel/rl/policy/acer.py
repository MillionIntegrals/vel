import typing
import gym
import torch
import torch.nn.functional as F

from vel.api import BackboneModule, ModuleFactory, BatchInfo, OptimizerFactory, VelOptimizer
from vel.metric.base import AveragingNamedMetric
from vel.rl.api import Trajectories, RlPolicy, Rollout
from vel.rl.module.q_stochastic_policy import QStochasticPolicy
from vel.util.situational import gym_space_to_size_hint


def select_indices(tensor, indices):
    """ Select indices from tensor """
    return tensor.gather(1, indices.unsqueeze(1)).squeeze()


class ACER(RlPolicy):
    """ Actor-Critic with Experience Replay - policy gradient calculations """

    def __init__(self, net: BackboneModule, target_net: typing.Optional[BackboneModule], action_space: gym.Space,
                 discount_factor: float, trust_region: bool = True, entropy_coefficient: float = 0.01,
                 q_coefficient: float = 0.5, rho_cap: float = 10.0, retrace_rho_cap: float = 1.0,
                 average_model_alpha: float = 0.99, trust_region_delta: float = 1.0):
        super().__init__(discount_factor)

        self.trust_region = trust_region

        self.entropy_coefficient = entropy_coefficient
        self.q_coefficient = q_coefficient

        self.rho_cap = rho_cap
        self.retrace_rho_cap = retrace_rho_cap

        # Trust region settings
        self.average_model_alpha = average_model_alpha
        self.trust_region_delta = trust_region_delta

        self.net = QStochasticPolicy(net, action_space)

        if self.trust_region:
            self.target_net = QStochasticPolicy(target_net, action_space)
            self.target_net.requires_grad_(False)
        else:
            self.target_net = None

    def create_optimizer(self, optimizer_factory: OptimizerFactory) -> VelOptimizer:
        """ Create optimizer for the purpose of optimizing this model """
        parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        return optimizer_factory.instantiate(parameters)

    def train(self, mode=True):
        """ Override train to make sure target model is always in eval mode """
        self.net.train(mode)

        if self.trust_region:
            self.target_net.train(False)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.net.reset_weights()

        if self.trust_region:
            self.target_net.load_state_dict(self.net.state_dict())

    def forward(self, observation, state=None):
        """ Calculate model outputs """
        return self.net(observation)

    def act(self, observation, state=None, deterministic=False):
        """ Select actions based on model's output """
        logprobs, q = self(observation)
        actions = self.net.action_head.sample(logprobs, deterministic=deterministic)

        # log likelihood of selected action
        action_logprobs = self.net.action_head.logprob(actions, logprobs)
        values = (torch.exp(logprobs) * q).sum(dim=1)

        return {
            'actions': actions,
            'q': q,
            'values': values,
            'action:logprobs': action_logprobs,
            'logprobs': logprobs
        }

    def update_target_policy(self):
        """ Update weights of the average model with new model observation """
        for model_param, average_param in zip(self.net.parameters(), self.target_net.parameters()):
            # EWMA average model update
            average_param.data.mul_(self.average_model_alpha).add_(model_param.data * (1 - self.average_model_alpha))

    def post_optimization_step(self, batch_info: BatchInfo, rollout: Rollout):
        """ Optional operations to perform after optimization """
        # We calculate the trust-region update with respect to the average model
        if self.trust_region:
            self.update_target_policy()

    def calculate_gradient(self, batch_info: BatchInfo, rollout: Rollout) -> dict:
        """ Calculate loss of the supplied rollout """
        assert isinstance(rollout, Trajectories), "ACER algorithm requires trajectory input"

        local_epsilon = 1e-6

        # Part 0.0 - Rollout values
        actions = rollout.batch_tensor('actions')
        rollout_probabilities = torch.exp(rollout.batch_tensor('logprobs'))
        observations = rollout.batch_tensor('observations')

        # PART 0.1 - Model evaluation
        logprobs, q = self(observations)

        # Selected action values
        action_logprobs = select_indices(logprobs, actions)
        action_q = select_indices(q, actions)

        # We only want to propagate gradients through specific variables
        with torch.no_grad():
            model_probabilities = torch.exp(logprobs)

            # Importance sampling correction - we must find the quotient of probabilities
            rho = model_probabilities / (rollout_probabilities + local_epsilon)

            # Probability quotient only for selected actions
            actions_rho = select_indices(rho, actions)

            # Calculate policy state values
            model_state_values = (model_probabilities * q).sum(dim=1)

            trajectory_rewards = rollout.transition_tensors['rewards']
            trajectory_dones = rollout.transition_tensors['dones']

            q_retraced = self.retrace(
                trajectory_rewards,
                trajectory_dones,
                action_q.reshape(trajectory_rewards.size()),
                model_state_values.reshape(trajectory_rewards.size()),
                actions_rho.reshape(trajectory_rewards.size()),
                rollout.rollout_tensors['final.values']
            ).flatten()

            advantages = q_retraced - model_state_values
            importance_sampling_coefficient = torch.min(actions_rho, self.rho_cap * torch.ones_like(actions_rho))

            explained_variance = 1 - torch.var(q_retraced - action_q) / torch.var(q_retraced)

        # Entropy of the policy distribution
        policy_entropy = torch.mean(self.net.action_head.entropy(logprobs))
        policy_gradient_loss = -torch.mean(advantages * importance_sampling_coefficient * action_logprobs)

        # Policy gradient bias correction
        with torch.no_grad():
            advantages_bias_correction = q - model_state_values.view(model_probabilities.size(0), 1)
            bias_correction_coefficient = F.relu(1.0 - self.rho_cap / (rho + local_epsilon))

        # This sum is an expectation with respect to action probabilities according to model policy
        policy_gradient_bias_correction_gain = torch.sum(
            logprobs * bias_correction_coefficient * advantages_bias_correction * model_probabilities,
            dim=1
        )

        policy_gradient_bias_correction_loss = - torch.mean(policy_gradient_bias_correction_gain)

        policy_loss = policy_gradient_loss + policy_gradient_bias_correction_loss

        q_function_loss = 0.5 * F.mse_loss(action_q, q_retraced)

        if self.trust_region:
            with torch.no_grad():
                target_logprobs = self.target_net(observations)[0]

            actor_loss = policy_loss - self.entropy_coefficient * policy_entropy
            q_loss = self.q_coefficient * q_function_loss

            actor_gradient = torch.autograd.grad(-actor_loss, logprobs, retain_graph=True)[0]

            # kl_divergence = model.kl_divergence(average_action_logits, action_logits).mean()
            # kl_divergence_grad = torch.autograd.grad(kl_divergence, action_logits, retain_graph=True)

            # Analytically calculated derivative of KL divergence on logits
            # That makes it hardcoded for discrete action spaces
            kl_divergence_grad_symbolic = - torch.exp(target_logprobs) / logprobs.size(0)

            k_dot_g = (actor_gradient * kl_divergence_grad_symbolic).sum(dim=-1)
            k_dot_k = (kl_divergence_grad_symbolic ** 2).sum(dim=-1)

            adjustment = (k_dot_g - self.trust_region_delta) / k_dot_k
            adjustment_clipped = adjustment.clamp(min=0.0)

            actor_gradient_updated = actor_gradient - adjustment_clipped.view(adjustment_clipped.size(0), 1)

            # Populate gradient from the newly updated fn
            logprobs.backward(gradient=-actor_gradient_updated, retain_graph=True)
            q_loss.backward(retain_graph=True)
        else:
            # Just populate gradient from the loss
            loss = policy_loss + self.q_coefficient * q_function_loss - self.entropy_coefficient * policy_entropy

            loss.backward()

        return {
            'policy_loss': policy_loss.item(),
            'policy_gradient_loss': policy_gradient_loss.item(),
            'policy_gradient_bias_correction': policy_gradient_bias_correction_loss.item(),
            'avg_q_selected': action_q.mean().item(),
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
        rho_bar = torch.min(torch.ones_like(rho) * self.retrace_rho_cap, rho)

        q_retraced_buffer = torch.zeros_like(rewards)

        next_value = final_values

        for i in reversed(range(rewards.size(0))):
            q_retraced = rewards[i] + self.discount_factor * next_value * (1.0 - dones[i])

            # Next iteration
            next_value = rho_bar[i] * (q_retraced - q_values[i]) + state_values[i]

            q_retraced_buffer[i] = q_retraced

        return q_retraced_buffer

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
            AveragingNamedMetric("model_prob_std"),
            AveragingNamedMetric("rollout_prob_std"),
            AveragingNamedMetric("avg_q_selected"),
            AveragingNamedMetric("avg_q_retraced")
        ]


class ACERFactory(ModuleFactory):
    """ Factory class for ACER policies """
    def __init__(self, net_factory, trust_region: bool, entropy_coefficient: float, q_coefficient: float,
                 discount_factor: float, rho_cap: float = 10.0, retrace_rho_cap: float = 1.0,
                 average_model_alpha: float = 0.99, trust_region_delta: float = 1.0):
        self.net_factory = net_factory
        self.trust_region = trust_region
        self.entropy_coefficient = entropy_coefficient
        self.q_coefficient = q_coefficient
        self.discount_factor = discount_factor
        self.rho_cap = rho_cap
        self.retrace_rho_cap = retrace_rho_cap
        self.average_model_alpha = average_model_alpha
        self.trust_region_delta = trust_region_delta

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        action_space = extra_args.pop('action_space')
        observation_space = extra_args.pop('observation_space')

        size_hint = gym_space_to_size_hint(observation_space)

        net = self.net_factory.instantiate(size_hint=size_hint, **extra_args)

        if self.trust_region:
            target_net = self.net_factory.instantiate(size_hint=size_hint, **extra_args)
        else:
            target_net = None

        return ACER(
            net=net,
            target_net=target_net,
            action_space=action_space,
            trust_region=self.trust_region,
            entropy_coefficient=self.entropy_coefficient,
            q_coefficient=self.q_coefficient,
            discount_factor=self.discount_factor,
            rho_cap=self.rho_cap,
            retrace_rho_cap=self.retrace_rho_cap,
            average_model_alpha=self.average_model_alpha,
            trust_region_delta=self.trust_region_delta,
        )


def create(net, trust_region: bool, entropy_coefficient: float, q_coefficient: float, discount_factor: float,
           rho_cap: float = 10.0, retrace_rho_cap: float = 1.0, average_model_alpha: float = 0.99,
           trust_region_delta: float = 1.0):
    """ Vel factory function """
    return ACERFactory(
        net_factory=net,
        trust_region=trust_region,
        entropy_coefficient=entropy_coefficient,
        q_coefficient=q_coefficient,
        rho_cap=rho_cap,
        retrace_rho_cap=retrace_rho_cap,
        discount_factor=discount_factor,
        average_model_alpha=average_model_alpha,
        trust_region_delta=trust_region_delta
    )
