import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils

from vel.api.metrics.averaging_metric import AveragingNamedMetric
from vel.math.functions import explained_variance
from vel.rl.api import AlgoBase, Rollout, Trajectories
from vel.rl.discount_bootstrap import discount_bootstrap_gae


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
        Avp = matrix_vector_operator(p)
        alpha = rdotr / torch.dot(p, Avp)

        x += alpha * p
        r -= alpha * Avp

        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr

        if rdotr < rdotr_tol:
            break

    return x


class TrpoPolicyGradient(AlgoBase):
    """ Trust Region Policy Optimization - https://arxiv.org/abs/1502.05477 """

    def __init__(self, max_kl, cg_iters, line_search_iters, cg_damping, entropy_coef, vf_iters,
                 discount_factor, gae_lambda, improvement_acceptance_ratio, max_grad_norm):
        self.mak_kl = max_kl
        self.cg_iters = cg_iters
        self.line_search_iters = line_search_iters
        self.cg_damping = cg_damping
        self.entropy_coef = entropy_coef
        self.vf_iters = vf_iters
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.improvement_acceptance_ratio = improvement_acceptance_ratio
        self.max_grad_norm = max_grad_norm

    def process_rollout(self, batch_info, rollout: Rollout):
        """ Process rollout for ALGO before any chunking/shuffling  """
        assert isinstance(rollout, Trajectories), "TRPO requires trajectory rollouts"

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

    def optimizer_step(self, batch_info, device, model, rollout):
        """ Single optimization step for a model """
        rollout = rollout.to_transitions()

        # This algorithm makes quote strong assumptions about how does the model look
        # so it does not make that much sense to switch to the evaluator interface
        # As it would be more of a problem than actual benefit

        observations = rollout.batch_tensor('observations')
        returns = rollout.batch_tensor('returns')

        # Evaluate model on the observations
        policy_params = model.policy(observations)
        policy_entropy = torch.mean(model.entropy(policy_params))

        policy_loss = self.calc_policy_loss(model, policy_params, policy_entropy, rollout)
        policy_grad = p2v(autograd.grad(policy_loss, model.policy_parameters(), retain_graph=True)).detach()

        # Calculate gradient of KL divergence of model with fixed version of itself
        # Value of kl_divergence will be 0, but what we need is the gradient, actually the 2nd derivarive
        kl_divergence = torch.mean(model.kl_divergence(policy_params.detach(), policy_params))
        kl_divergence_gradient = p2v(torch.autograd.grad(kl_divergence, model.policy_parameters(), create_graph=True))

        step_direction = conjugate_gradient_method(
            matrix_vector_operator=lambda x: self.fisher_vector_product(x, kl_divergence_gradient, model),
            # Because we want to decrease the loss, we want to go into the direction of -gradient
            loss_gradient=-policy_grad,
            nsteps=self.cg_iters
        )

        shs = 0.5 * step_direction @ self.fisher_vector_product(step_direction, kl_divergence_gradient, model)
        lm = torch.sqrt(shs / self.mak_kl)
        full_step = step_direction / lm

        # Because we want to decrease the loss, we want to go into the direction of -gradient
        expected_improvement = (-policy_grad) @ full_step
        original_parameter_vec = p2v(model.policy_parameters()).detach_()

        policy_optimization_success, ratio, policy_loss_improvement, new_policy_loss, kl_divergence_step = self.line_search(
            model, rollout, policy_loss, policy_params, original_parameter_vec, full_step, expected_improvement
        )

        gradient_norms = []

        for i in range(self.vf_iters):
            batch_info.optimizer.zero_grad()
            value_loss = self.value_loss(model, observations, returns)

            value_loss.backward()

            # Gradient clipping
            if self.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=self.max_grad_norm
                )

                gradient_norms.append(grad_norm)

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

    def line_search(self, model, rollout, original_policy_loss, original_policy_params, original_parameter_vec,
                    full_step, expected_improvement_full):
        """ Find the right stepsize to make sure policy improves """
        current_parameter_vec = original_parameter_vec.clone()

        for idx in range(self.line_search_iters):
            stepsize = 0.5 ** idx

            new_parameter_vec = current_parameter_vec + stepsize * full_step

            # Update model parameters
            v2p(new_parameter_vec, model.policy_parameters())

            # Calculate new loss
            with torch.no_grad():
                policy_params = model.policy(rollout.batch_tensor('observations'))
                policy_entropy = torch.mean(model.entropy(policy_params))
                kl_divergence = torch.mean(model.kl_divergence(original_policy_params, policy_params))

                new_loss = self.calc_policy_loss(model, policy_params, policy_entropy, rollout)

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
        v2p(original_parameter_vec, model.policy_parameters())
        return False, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    def fisher_vector_product(self, vector, kl_divergence_gradient, model):
        """ Calculate product Hessian @ vector """
        assert not vector.requires_grad, "Vector must not propagate gradient"
        dot_product = vector @ kl_divergence_gradient

        # at least one dimension spans across two contiguous subspaces
        double_gradient = torch.autograd.grad(dot_product, model.policy_parameters(), retain_graph=True)
        fvp = p2v(x.contiguous() for x in double_gradient)

        return fvp + vector * self.cg_damping

    def value_loss(self, model, observations, discounted_rewards):
        """ Loss of value estimator """
        value_outputs = model.value(observations)
        value_loss = 0.5 * F.mse_loss(value_outputs, discounted_rewards)
        return value_loss

    def calc_policy_loss(self, model, policy_params, policy_entropy, rollout):
        """
        Policy gradient loss - calculate from probability distribution

        Calculate surrogate loss - advantage * policy_probability / fixed_initial_policy_probability

        Because we operate with logarithm of -probability (neglogp) we do
        - advantage * exp(fixed_neglogps - model_neglogps)
        """
        actions = rollout.batch_tensor('actions')
        advantages = rollout.batch_tensor('advantages')
        fixed_logprobs = rollout.batch_tensor('action:logprobs')

        model_logprobs = model.logprob(actions, policy_params)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # We put - in front because we want to maximize the surrogate objective
        policy_loss = -advantages * torch.exp(model_logprobs - fixed_logprobs)

        return policy_loss.mean() - policy_entropy * self.entropy_coef

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("new_policy_loss"),
            AveragingNamedMetric("policy_entropy"),
            AveragingNamedMetric("value_loss"),
            AveragingNamedMetric("policy_optimization_success"),
            AveragingNamedMetric("policy_improvement_ratio"),
            AveragingNamedMetric("kl_divergence_step"),
            AveragingNamedMetric("policy_loss_improvement"),
            AveragingNamedMetric("grad_norm"),
            AveragingNamedMetric("advantage_norm"),
            AveragingNamedMetric("explained_variance")
        ]


def create(max_kl, cg_iters, line_search_iters, cg_damping, entropy_coef, vf_iters, discount_factor,
           gae_lambda=1.0, improvement_acceptance_ratio=0.1, max_grad_norm=0.5):
    """ Vel factory function """
    return TrpoPolicyGradient(
        max_kl, int(cg_iters), int(line_search_iters), cg_damping, entropy_coef, vf_iters,
        discount_factor=discount_factor,
        gae_lambda=gae_lambda,
        improvement_acceptance_ratio=improvement_acceptance_ratio,
        max_grad_norm=max_grad_norm
    )
