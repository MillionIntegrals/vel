import typing

import gym
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

import vel.util.module_util as mu

from vel.api import BackboneNetwork, BatchInfo, ModelFactory, OptimizerFactory, VelOptimizer, SizeHints
from vel.metric.base import AveragingNamedMetric
from vel.rl.api import RlPolicy, Rollout
from vel.rl.module.actor_critic_policy import ActorCriticPolicy
from vel.rl.module.noise.ou_noise import OuNoise
from vel.util.situational import gym_space_to_size_hint


class DDPG(RlPolicy):
    """ Deep Deterministic Policy Gradient (DDPG) - policy gradient calculations """

    def __init__(self, net: BackboneNetwork, target_net: BackboneNetwork, action_space: gym.Space,
                 discount_factor: float, tau: float, noise_std_dev: float):
        super().__init__(discount_factor)

        self.net = net
        self.target_net = target_net

        self.tau = tau
        self.discount_factor = discount_factor

        self.action_noise = OuNoise(std_dev=noise_std_dev, action_space=action_space)

    def train(self, mode=True):
        """ Override train to make sure target model is always in eval mode """
        self.net.train(mode)
        self.target_net.train(False)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.net.reset_weights()
        self.target_net.load_state_dict(self.net.state_dict())

    def reset_episodic_state(self, dones: torch.Tensor):
        """ Called by the rollout worker, whenever episode is finished """
        self.action_noise.reset_episodic_state(dones)

    def create_optimizer(self, optimizer_factory: OptimizerFactory) -> VelOptimizer:
        """ Create optimizer for the purpose of optimizing this model """
        parameter_groups = mu.to_parameter_groups(self.net.layer_groups())
        return optimizer_factory.instantiate_parameter_groups(parameter_groups)

    def forward(self, observation, state=None):
        """ Calculate model outputs """
        return self.net(observation)

    def act(self, observation, state=None, deterministic=False) -> dict:
        """ Select actions based on model's output """
        action, value = self(observation)

        if deterministic:
            noisy_action = action
        else:
            noisy_action = self.action_noise(action)

        return {
            'actions': noisy_action,
            'values': value
        }

    def calculate_gradient(self, batch_info: BatchInfo, rollout: Rollout) -> dict:
        """ Calculate loss of the supplied rollout """
        rollout = rollout.to_transitions()

        dones = rollout.batch_tensor('dones')
        rewards = rollout.batch_tensor('rewards')
        observations_next = rollout.batch_tensor('observations_next')
        actions = rollout.batch_tensor('actions')
        observations = rollout.batch_tensor('observations')

        # Calculate value loss - or critic loss
        with torch.no_grad():
            target_next_value = self.target_net.value(observations_next)
            target_value = rewards + (1.0 - dones) * self.discount_factor * target_next_value

        # Value estimation error vs the target network
        model_value = self.net.value(observations, actions)
        value_loss = F.mse_loss(model_value, target_value)

        # It may seem a bit tricky what I'm doing here, but the underlying idea is simple
        # All other implementations I found keep two separate optimizers for actor and critic
        # and update them separately
        # What I'm trying to do is to optimize them both with a single optimizer
        # but I need to make sure gradients flow correctly
        # From critic loss to critic network only and from actor loss to actor network only

        # Backpropagate value loss to critic only
        value_loss.backward()

        model_action = self.net.action(observations)
        model_action_value = self.net.value(observations, model_action)

        policy_loss = -model_action_value.mean()

        model_action_grad = torch.autograd.grad(policy_loss, model_action)[0]

        # Backpropagate actor loss to actor only
        model_action.backward(gradient=model_action_grad)

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

    def post_optimization_step(self, batch_info: BatchInfo, rollout: Rollout):
        """ Steps to take after optimization has been done"""
        # Update target model
        for model_param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
            # EWMA average model update
            target_param.data.mul_(1 - self.tau).add_(model_param.data * self.tau)

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("value_loss"),
            AveragingNamedMetric("policy_loss"),
        ]


class DDPGFactory(ModelFactory):
    """ Factory for the DDPG policy """

    def __init__(self, actor_net: ModelFactory, critic_net: ModelFactory,
                 discount_factor: float, tau: float, noise_std_dev: float,
                 input_net: typing.Optional[ModelFactory] = None):
        self.actor_net_factory = actor_net
        self.critic_net_factory = critic_net
        self.input_net_factory = input_net

        self.discount_factor = discount_factor
        self.tau = tau
        self.noise_std_dev = noise_std_dev

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        action_space = extra_args.pop('action_space')
        observation_space = extra_args.pop('observation_space')

        size_hint = gym_space_to_size_hint(observation_space)
        action_hint = gym_space_to_size_hint(action_space)

        if self.input_net_factory is None:
            target_input_net = input_net = nn.Identity()
        else:
            input_net = self.input_net_factory.instantiate(size_hint=size_hint, **extra_args)
            target_input_net = self.input_net_factory.instantiate(size_hint=size_hint, **extra_args)
            size_hint = input_net.size_hints()

        critic_size_hint = SizeHints((size_hint.unwrap(), action_hint.unwrap()))

        actor_net = self.actor_net_factory.instantiate(size_hint=size_hint, **extra_args)
        critic_net = self.critic_net_factory.instantiate(size_hint=critic_size_hint, **extra_args)

        net = ActorCriticPolicy(
            input_net, actor_net, critic_net, action_space
        )

        target_actor_net = self.actor_net_factory.instantiate(size_hint=size_hint, **extra_args)
        target_critic_net = self.critic_net_factory.instantiate(size_hint=critic_size_hint, **extra_args)

        target_net = ActorCriticPolicy(
            target_input_net, target_actor_net, target_critic_net, action_space
        )

        return DDPG(
            net=net,
            target_net=target_net,
            action_space=action_space,
            discount_factor=self.discount_factor,
            tau=self.tau,
            noise_std_dev=self.noise_std_dev
        )


def create(actor_net: ModelFactory, critic_net: ModelFactory,
           discount_factor: float, tau: float, noise_std_dev: float,
           input_net: typing.Optional[ModelFactory] = None
           ):
    """ Vel factory function """
    return DDPGFactory(
        actor_net=actor_net,
        critic_net=critic_net,
        input_net=input_net,
        discount_factor=discount_factor,
        tau=tau,
        noise_std_dev=noise_std_dev
    )
