import numbers

import typing
import gym
import torch
import torch.nn.functional as F
import torch.nn.utils

from vel.api import ModelFactory, BackboneNetwork, BatchInfo, Schedule
from vel.function.constant import ConstantSchedule
from vel.metric import AveragingNamedMetric
from vel.rl.api import RlPolicy, Rollout
from vel.rl.module.q_policy import QPolicy
from vel.rl.module.noise.eps_greedy import EpsGreedy


class DQN(RlPolicy):
    """ Deep Q-Learning algorithm """

    def __init__(self, net: BackboneNetwork, net_factory: ModelFactory, action_space: gym.Space,
                 epsilon: typing.Union[float, Schedule], discount_factor: float, double_dqn: bool,
                 dueling_dqn: bool, target_update_frequency: int):
        super().__init__(discount_factor)

        self.model = QPolicy(net=net, action_space=action_space, dueling_dqn=dueling_dqn)

        self.double_dqn = double_dqn
        self.target_update_frequency = target_update_frequency

        if isinstance(epsilon, numbers.Number):
            self.epsilon_schedule = ConstantSchedule(epsilon)
        else:
            self.epsilon_schedule = epsilon

        self.epsilon_value = self.epsilon_schedule.value(0.0)

        self.action_noise = EpsGreedy(action_space=action_space)

        self.target_model = QPolicy(net=net_factory.instantiate(), action_space=action_space, dueling_dqn=dueling_dqn)

    def train(self, mode=True):
        """ Override train to make sure target model is always in eval mode """
        self.model.train(mode)
        self.target_model.train(False)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.model.reset_weights()
        self.target_model.load_state_dict(self.model.state_dict())

    def forward(self, observation, state=None):
        """ Calculate model outputs """
        return self.model(observation)

    def act(self, observation, state=None, deterministic=False):
        """ Select actions based on model's output """
        q_values = self.model(observation)
        actions = self.model.q_head.sample(q_values)
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

        q = self.model(observations)

        with torch.no_grad():
            target_q = self.target_model(observations_next)

            if self.double_dqn:
                # DOUBLE DQN
                model_q_next = self.model(observations_next)
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
            self.target_model.load_state_dict(self.model.state_dict())

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
        net = self.net_factory.instantiate(**extra_args)

        return DQN(
            net=net,
            net_factory=self.net_factory,
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
