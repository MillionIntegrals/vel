import numbers

import typing
import gym
import torch
import torch.nn.functional as F
import torch.nn.utils

from vel.api import ModelFactory, BackboneNetwork, BatchInfo, Schedule, OptimizerFactory, VelOptimizer
from vel.function.constant import ConstantSchedule
from vel.metric import AveragingNamedMetric
from vel.rl.api import RlPolicy, Rollout
from vel.rl.module.q_policy import QPolicy
from vel.rl.module.noise.eps_greedy import EpsGreedy
from vel.util.situational import gym_space_to_size_hint


class DQN(RlPolicy):
    """ Deep Q-Learning algorithm """

    def __init__(self, net: BackboneNetwork, target_net: BackboneNetwork, action_space: gym.Space,
                 epsilon: typing.Union[float, Schedule], discount_factor: float, double_dqn: bool,
                 dueling_dqn: bool, target_update_frequency: int):
        super().__init__(discount_factor)

        self.net = QPolicy(net=net, action_space=action_space, dueling_dqn=dueling_dqn)
        self.target_net = QPolicy(net=target_net, action_space=action_space, dueling_dqn=dueling_dqn)
        self.target_net.requires_grad_(False)

        self.double_dqn = double_dqn
        self.target_update_frequency = target_update_frequency

        if isinstance(epsilon, numbers.Number):
            self.epsilon_schedule = ConstantSchedule(epsilon)
        else:
            self.epsilon_schedule = epsilon

        self.epsilon_value = self.epsilon_schedule.value(0.0)
        self.action_noise = EpsGreedy(action_space=action_space)

    def create_optimizer(self, optimizer_factory: OptimizerFactory) -> VelOptimizer:
        """ Create optimizer for the purpose of optimizing this model """
        parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        return optimizer_factory.instantiate(parameters)

    def train(self, mode=True):
        """ Override train to make sure target model is always in eval mode """
        self.net.train(mode)
        self.target_net.train(False)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.net.reset_weights()
        self.target_net.load_state_dict(self.net.state_dict())

    def forward(self, observation, state=None):
        """ Calculate model outputs """
        return self.net(observation)

    def act(self, observation, state=None, deterministic=False):
        """ Select actions based on model's output """
        q_values = self.net(observation)
        actions = self.net.q_head.sample(q_values)
        noisy_actions = self.action_noise(actions, epsilon=self.epsilon_value, deterministic=deterministic)

        return {
            'actions': noisy_actions,
            'q': q_values
        }

    def calculate_gradient(self, batch_info: BatchInfo, rollout: Rollout) -> dict:
        """ Calculate loss of the supplied rollout """
        observations = rollout.batch_tensor('observations')
        observations_next = rollout.batch_tensor('observations_next')

        actions = rollout.batch_tensor('actions')
        dones_tensor = rollout.batch_tensor('dones')
        rewards_tensor = rollout.batch_tensor('rewards')

        assert dones_tensor.dtype == torch.float32

        q = self.net(observations)

        with torch.no_grad():
            target_q = self.target_net(observations_next)

            if self.double_dqn:
                # DOUBLE DQN
                model_q_next = self.net(observations_next)
                # Select largest 'target' value based on action that 'model' selects
                values = target_q.gather(1, model_q_next.argmax(dim=1, keepdim=True)).squeeze(1)
            else:
                # REGULAR DQN
                # [0] is because in pytorch .max(...) returns tuple (max values, argmax)
                values = target_q.max(dim=1)[0]

            forward_steps = rollout.extra_data.get('forward_steps', 1)
            estimated_return = rewards_tensor + (self.discount_factor ** forward_steps) * values * (1 - dones_tensor)

        q_selected = q.gather(1, actions.unsqueeze(1)).squeeze(1)

        if rollout.has_tensor('weights'):
            weights = rollout.batch_tensor('weights')
        else:
            weights = torch.ones_like(rewards_tensor)

        original_losses = F.smooth_l1_loss(q_selected, estimated_return, reduction='none')

        loss_value = torch.mean(weights * original_losses)
        loss_value.backward()

        return {
            'loss': loss_value.item(),
            # We need it to update priorities in the replay buffer:
            'errors': original_losses.detach().cpu().numpy(),
            'average_q_selected': torch.mean(q_selected).item(),
            'average_q_target': torch.mean(estimated_return).item()
        }

    def post_optimization_step(self, batch_info, rollout):
        """ Steps to take after optimization has been done"""
        if batch_info.aggregate_batch_number % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.epsilon_value = self.epsilon_schedule.value(batch_info['progress'])

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("loss", scope="model"),
            AveragingNamedMetric("average_q_selected", scope="model"),
            AveragingNamedMetric("average_q_target", scope="model")
        ]


class DQNFactory(ModelFactory):
    def __init__(self, net_factory: ModelFactory, epsilon: typing.Union[float, Schedule], discount_factor: float,
                 target_update_frequency: int, double_dqn: bool = False, dueling_dqn: bool = False):
        self.net_factory = net_factory
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.target_update_frequency = target_update_frequency
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        action_space = extra_args.pop('action_space')
        observation_space = extra_args.pop('observation_space')

        size_hint = gym_space_to_size_hint(observation_space)

        net = self.net_factory.instantiate(size_hint=size_hint, **extra_args)
        target_net = self.net_factory.instantiate(size_hint=size_hint, **extra_args)

        return DQN(
            net=net,
            target_net=target_net,
            action_space=action_space,
            epsilon=self.epsilon,
            discount_factor=self.discount_factor,
            double_dqn=self.double_dqn,
            dueling_dqn=self.dueling_dqn,
            target_update_frequency=self.target_update_frequency
        )


def create(net: ModelFactory, epsilon: typing.Union[float, Schedule], discount_factor: float,
           target_update_frequency: int, double_dqn: bool = False, dueling_dqn: bool = False):
    """ Vel factory function """

    return DQNFactory(
        net_factory=net,
        epsilon=epsilon,
        discount_factor=discount_factor,
        double_dqn=double_dqn,
        dueling_dqn=dueling_dqn,
        target_update_frequency=target_update_frequency,
    )
