import gym
import numpy as np
import itertools as it

import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils
import typing

from vel.api import BatchInfo, VelOptimizer, BackboneModel, LinearBackboneModel, OptimizerFactory, ModelFactory
from vel.calc.function import explained_variance
from vel.metric.base import AveragingNamedMetric
from vel.module.input.identity import IdentityFactory

from vel.rl.api import Rollout, Trajectories, RlPolicy
from vel.rl.discount_bootstrap import discount_bootstrap_gae
from vel.rl.module.stochastic_action_head import StochasticActionHead
from vel.rl.module.value_head import ValueHead


def p2v(params):
    """ Parameters to vector - shorthand utility version """
    return torch.nn.utils.parameters_to_vector(params)


def v2p(vector, params):
    """ Vector to parameters - shorthand utility version """
    return torch.nn.utils.vector_to_parameters(vector, params)


def conjugate_gradient_method(matrix_vector_operator, loss_gradient, nsteps, rdotr_tol=1e-10):
    """ Conjugate gradient algorithm """
    x = torch.zeros_like(loss_gradient)

    r = loss_gradient.clone()
    p = loss_gradient.clone()

    rdotr = torch.dot(r, r)

    for i in range(nsteps):
        avp = matrix_vector_operator(p)
        alpha = rdotr / torch.dot(p, avp)

        x += alpha * p
        r -= alpha * avp

        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr

        if rdotr < rdotr_tol:
            break

    return x


class TRPO(RlPolicy):
    """ Trust Region Policy Optimization - https://arxiv.org/abs/1502.05477 """

    def __init__(self,
                 input_block: BackboneModel,
                 policy_backbone: LinearBackboneModel, value_backbone: LinearBackboneModel,
                 action_space: gym.Space,
                 max_kl, cg_iters, line_search_iters, cg_damping, entropy_coefficient, vf_iters,
                 discount_factor, gae_lambda, improvement_acceptance_ratio):
        super().__init__(discount_factor)

        self.input_block = input_block
        self.policy_backbone = policy_backbone
        self.value_backbone = value_backbone

        self.action_head = StochasticActionHead(
            action_space=action_space,
            input_dim=self.policy_backbone.output_dim
        )

        self.value_head = ValueHead(input_dim=self.value_backbone.output_dim)

        self.mak_kl = max_kl
        self.cg_iters = cg_iters
        self.line_search_iters = line_search_iters
        self.cg_damping = cg_damping
        self.entropy_coefficient = entropy_coefficient
        self.vf_iters = vf_iters
        self.gae_lambda = gae_lambda
        self.improvement_acceptance_ratio = improvement_acceptance_ratio

    def reset_weights(self):
        """ Initialize properly model weights """
        self.input_block.reset_weights()

        self.policy_backbone.reset_weights()
        self.value_backbone.reset_weights()

        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observations):
        """ Calculate model outputs """
        input_data = self.input_block(observations)

        policy_base_output = self.policy_backbone(input_data)
        value_base_output = self.value_backbone(input_data)

        action_output = self.action_head(policy_base_output)
        value_output = self.value_head(value_base_output)

        return action_output, value_output

    def value(self, observations, state=None):
        """ Calculate only value head for given state """
        input_data = self.input_block(observations)
        base_output = self.value_backbone(input_data)
        value_output = self.value_head(base_output)
        return value_output

    def policy(self, observations):
        """ Calculate only action head for given state """
        input_data = self.input_block(observations)
        policy_base_output = self.policy_backbone(input_data)
        policy_params = self.action_head(policy_base_output)
        return policy_params

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

    def create_optimizer(self, optimizer_factory: OptimizerFactory) -> VelOptimizer:
        """ Create optimizer for the purpose of optimizing this model """
        parameters = filter(lambda p: p.requires_grad, self.value_parameters())
        return optimizer_factory.instantiate(parameters)

    def policy_parameters(self):
        """ Parameters of policy """
        return it.chain(
            self.input_block.parameters(),
            self.policy_backbone.parameters(),
            self.action_head.parameters()
        )

    def value_parameters(self):
        """ Parameters of value function """
        return it.chain(
            self.input_block.parameters(),
            self.value_backbone.parameters(),
            self.value_head.parameters()
        )

    def process_rollout(self, rollout: Rollout):
        """ Process rollout for optimization before any chunking/shuffling  """
        assert isinstance(rollout, Trajectories), "PPO requires trajectory rollouts"

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

    def optimize(self, batch_info: BatchInfo, rollout: Rollout) -> dict:
        """ Single optimization step for a model """
        rollout = rollout.to_transitions()

        observations = rollout.batch_tensor('observations')
        returns = rollout.batch_tensor('returns')

        # Evaluate model on the observations
        action_pd_params = self.policy(observations)
        policy_entropy = torch.mean(self.action_head.entropy(action_pd_params))

        policy_loss = self.calc_policy_loss(action_pd_params, policy_entropy, rollout)
        policy_grad = p2v(autograd.grad(policy_loss, self.policy_parameters(), retain_graph=True)).detach()

        # Calculate gradient of KL divergence of model with fixed version of itself
        # Value of kl_divergence will be 0, but what we need is the gradient, actually the 2nd derivarive
        kl_divergence = torch.mean(self.action_head.kl_divergence(action_pd_params.detach(), action_pd_params))
        kl_divergence_gradient = p2v(torch.autograd.grad(kl_divergence, self.policy_parameters(), create_graph=True))

        step_direction = conjugate_gradient_method(
            matrix_vector_operator=lambda x: self.fisher_vector_product(x, kl_divergence_gradient),
            # Because we want to decrease the loss, we want to go into the direction of -gradient
            loss_gradient=-policy_grad,
            nsteps=self.cg_iters
        )

        shs = 0.5 * step_direction @ self.fisher_vector_product(step_direction, kl_divergence_gradient)
        lm = torch.sqrt(shs / self.mak_kl)
        full_step = step_direction / lm

        # Because we want to decrease the loss, we want to go into the direction of -gradient
        expected_improvement = (-policy_grad) @ full_step
        original_parameter_vec = p2v(self.policy_parameters()).detach_()

        (policy_optimization_success, ratio, policy_loss_improvement, new_policy_loss, kl_divergence_step) = (
            self.line_search(
                rollout, policy_loss, action_pd_params, original_parameter_vec, full_step, expected_improvement
            )
        )

        gradient_norms = []

        for i in range(self.vf_iters):
            batch_info.optimizer.zero_grad()
            value_loss = self.value_loss(observations, returns)

            value_loss.backward()

            batch_info.optimizer.step(closure=None)

        if gradient_norms:
            gradient_norm = np.mean(gradient_norms)
        else:
            gradient_norm = 0.0

        # noinspection PyUnboundLocalVariable
        return {
            'new_policy_loss': new_policy_loss.item(),
            'policy_entropy': policy_entropy.item(),
            'value_loss': value_loss.item(),
            'policy_optimization_success': float(policy_optimization_success),
            'policy_improvement_ratio': ratio.item(),
            'kl_divergence_step': kl_divergence_step.item(),
            'policy_loss_improvement': policy_loss_improvement.item(),
            'grad_norm': gradient_norm,
            'advantage_norm': torch.norm(rollout.batch_tensor('advantages')).item(),
            'explained_variance': explained_variance(returns, rollout.batch_tensor('values'))
        }

    def line_search(self, rollout, original_policy_loss, original_policy_params, original_parameter_vec,
                    full_step, expected_improvement_full):
        """ Find the right stepsize to make sure policy improves """
        current_parameter_vec = original_parameter_vec.clone()

        for idx in range(self.line_search_iters):
            stepsize = 0.5 ** idx

            new_parameter_vec = current_parameter_vec + stepsize * full_step

            # Update model parameters
            v2p(new_parameter_vec, self.policy_parameters())

            # Calculate new loss
            with torch.no_grad():
                policy_params = self.policy(rollout.batch_tensor('observations'))
                policy_entropy = torch.mean(self.action_head.entropy(policy_params))
                kl_divergence = torch.mean(self.action_head.kl_divergence(original_policy_params, policy_params))

                new_loss = self.calc_policy_loss(policy_params, policy_entropy, rollout)

                actual_improvement = original_policy_loss - new_loss
                expected_improvement = expected_improvement_full * stepsize

                ratio = actual_improvement / expected_improvement

            if kl_divergence.item() > self.mak_kl * 1.5:
                # KL divergence bound exceeded
                continue
            elif ratio < expected_improvement:
                # Not enough loss improvement
                continue
            else:
                # Optimization successful
                return True, ratio, actual_improvement, new_loss, kl_divergence

        # Optimization failed, revert to initial parameters
        v2p(original_parameter_vec, self.policy_parameters())
        return False, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    def fisher_vector_product(self, vector, kl_divergence_gradient):
        """ Calculate product Hessian @ vector """
        assert not vector.requires_grad, "Vector must not propagate gradient"
        dot_product = vector @ kl_divergence_gradient

        # at least one dimension spans across two contiguous subspaces
        double_gradient = torch.autograd.grad(dot_product, self.policy_parameters(), retain_graph=True)
        fvp = p2v(x.contiguous() for x in double_gradient)

        return fvp + vector * self.cg_damping

    def value_loss(self, observations, returns):
        """ Loss of value function head """
        value_outputs = self.value(observations)
        value_loss = 0.5 * F.mse_loss(value_outputs, returns)
        return value_loss

    def calc_policy_loss(self, policy_params, policy_entropy, rollout):
        """
        Policy gradient loss - calculate from probability distribution

        Calculate surrogate loss - advantage * policy_probability / fixed_initial_policy_probability

        Because we operate with logarithm of -probability (neglogp) we do
        - advantage * exp(fixed_neglogps - model_neglogps)
        """
        actions = rollout.batch_tensor('actions')
        advantages = rollout.batch_tensor('advantages')
        fixed_logprobs = rollout.batch_tensor('action:logprobs')

        model_logprobs = self.action_head.logprob(actions, policy_params)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # We put - in front because we want to maximize the surrogate objective
        policy_loss = -advantages * torch.exp(model_logprobs - fixed_logprobs)

        return policy_loss.mean() - policy_entropy * self.entropy_coefficient

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("new_policy_loss", scope="model"),
            AveragingNamedMetric("policy_entropy", scope="model"),
            AveragingNamedMetric("value_loss", scope="model"),
            AveragingNamedMetric("policy_optimization_success", scope="model"),
            AveragingNamedMetric("policy_improvement_ratio", scope="model"),
            AveragingNamedMetric("kl_divergence_step", scope="model"),
            AveragingNamedMetric("policy_loss_improvement", scope="model"),
            AveragingNamedMetric("advantage_norm", scope="model"),
            AveragingNamedMetric("explained_variance", scope="model")
        ]


class TRPOFactory(ModelFactory):
    """ Factory class for policy gradient models """
    def __init__(self, input_block, policy_backbone: ModelFactory, value_backbone: ModelFactory,
                 max_kl, cg_iters, line_search_iters, cg_damping, entropy_coefficient, vf_iters,
                 discount_factor, gae_lambda, improvement_acceptance_ratio):
        self.policy_backbone = policy_backbone
        self.value_backbone = value_backbone
        self.input_block = input_block
        self.entropy_coefficient = entropy_coefficient

        self.mak_kl = max_kl
        self.cg_iters = cg_iters
        self.line_search_iters = line_search_iters
        self.cg_damping = cg_damping
        self.vf_iters = vf_iters
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.improvement_acceptance_ratio = improvement_acceptance_ratio

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        action_space = extra_args.pop('action_space')

        input_block = self.input_block.instantiate()

        policy_backbone = self.policy_backbone.instantiate(**extra_args)
        value_backbone = self.value_backbone.instantiate(**extra_args)

        return TRPO(
            input_block=input_block,
            policy_backbone=policy_backbone,
            value_backbone=value_backbone,
            action_space=action_space,
            max_kl=self.mak_kl,
            cg_iters=self.cg_iters,
            line_search_iters=self.line_search_iters,
            cg_damping=self.cg_damping,
            entropy_coefficient=self.entropy_coefficient,
            vf_iters=self.vf_iters,
            discount_factor=self.discount_factor,
            gae_lambda=self.gae_lambda,
            improvement_acceptance_ratio=self.improvement_acceptance_ratio
        )


def create(policy_backbone: ModelFactory, value_backbone: ModelFactory,
           max_kl, cg_iters, line_search_iters, cg_damping, entropy_coefficient, vf_iters,
           discount_factor, gae_lambda, improvement_acceptance_ratio,
           input_block: typing.Optional[ModelFactory] = None):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return TRPOFactory(
        input_block=input_block,
        policy_backbone=policy_backbone,
        value_backbone=value_backbone,
        max_kl=max_kl,
        cg_iters=cg_iters,
        line_search_iters=line_search_iters,
        cg_damping=cg_damping,
        entropy_coefficient=entropy_coefficient,
        vf_iters=vf_iters,
        discount_factor=discount_factor,
        gae_lambda=gae_lambda,
        improvement_acceptance_ratio=improvement_acceptance_ratio,
    )

