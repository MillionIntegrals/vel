import numpy as np
import torch

from vel.api.base import Schedule
from vel.api.metrics import AveragingNamedMetric
from vel.rl.api.base import ReplayEnvRollerBase, EnvRollerFactory
from vel.rl.buffers.prioritized_backend import PrioritizedReplayBackend


class PrioritizedReplayRollerEpsGreedy(ReplayEnvRollerBase):
    """
    Environment roller for action-value models using experience replay.
    Experience replay buffer implementation with prioritized sampling based on td-errors

    Because framestack is implemented directly in the buffer, we can use *much* less space to hold samples in
    memory for very little additional cost.
    """

    def __init__(self, environment, device, epsilon_schedule: Schedule, batch_size: int,
                 buffer_capacity: int, buffer_initial_size: int, frame_stack: int,
                 priority_exponent: float, priority_weight: Schedule, priority_epsilon: float):
        self.epsilon_schedule = epsilon_schedule

        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack = frame_stack

        self.priority_exponent = priority_exponent
        self.priority_weight_schedule = priority_weight
        self.priority_epsilon = priority_epsilon

        self._environment = environment
        self.device = device

        self.backend = PrioritizedReplayBackend(
            buffer_capacity=self.buffer_capacity,
            observation_space=environment.observation_space,
            action_space=environment.action_space
        )

        self.last_observation = self.environment.reset()

    @property
    def environment(self):
        """ Return environment of this env roller """
        return self._environment

    def is_ready_for_sampling(self) -> bool:
        """ If buffer is ready for drawing samples from it (usually checks if there is enough data) """
        return self.backend.current_size >= self.buffer_initial_size

    def epsgreedy_action(self, policy_action, epsilon):
        """ Sample e-greedy action using curreny policy and epsilon value """
        random_samples = torch.randint_like(policy_action, high=self.environment.action_space.n)
        selector = torch.rand_like(policy_action, dtype=torch.float32)
        return torch.where(selector > epsilon, policy_action, random_samples)

    def rollout(self, batch_info, model) -> dict:
        """ Roll-out the environment and return it """
        epsilon_value = self.epsilon_schedule.value(batch_info['progress'])
        batch_info['epsilon'] = epsilon_value

        last_observation = np.concatenate([
            self.backend.get_frame(self.backend.current_idx, self.frame_stack - 1),
            self.last_observation
        ], axis=-1)

        observation_tensor = torch.from_numpy(last_observation[None]).to(self.device)

        step = model.step(observation_tensor)
        epsgreedy_step = self.epsgreedy_action(step['actions'], epsilon_value)
        action = epsgreedy_step.item()

        observation, reward, done, info = self.environment.step(action)

        self.backend.store_transition(self.last_observation, action, reward, done)

        # Usual, reset on done
        if done:
            observation = self.environment.reset()

        self.last_observation = observation

        return {
            'epsilon': epsilon_value,
            'episode_information': info.get('episode'),
            'action': epsgreedy_step[0],
            'value': step['values'][0]
        }

    def metrics(self):
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("epsilon"),
        ]

    def sample(self, batch_info, model) -> dict:
        """ Sample experience from replay buffer and return a batch """
        probs, indexes, tree_idxs = self.backend.sample_batch_prioritized(self.batch_size, self.frame_stack)
        batch = self.backend.get_batch(indexes, self.frame_stack)

        # Normalize weights properly
        priority_weight = self.priority_weight_schedule.value(batch_info['progress'])

        probs = np.stack(probs) / self.backend.segment_tree.total()
        capacity = self.backend.deque.current_size
        weights = (capacity * probs) ** (-priority_weight)
        weights = weights / weights.max()

        observations = torch.from_numpy(batch['states']).to(self.device)
        observations_plus1 = torch.from_numpy(batch['states+1']).to(self.device)
        dones = torch.from_numpy(batch['dones'].astype(np.float32)).to(self.device)
        rewards = torch.from_numpy(batch['rewards'].astype(np.float32)).to(self.device)
        actions = torch.from_numpy(batch['actions']).to(self.device)
        weights = torch.from_numpy(weights.astype(np.float32)).to(self.device)

        return {
            'size': self.batch_size,
            'observations': observations,
            'observations+1': observations_plus1,
            'dones': dones,
            'rewards': rewards,
            'actions': actions,
            'weights': weights,
            'tree_idxs': tree_idxs
        }

    def update(self, sample, batch_info):
        """ Update sample weights in the priority buffer """
        errors = batch_info['errors']

        weights = (errors + self.priority_epsilon) ** self.priority_exponent

        for idx, priority in zip(sample['tree_idxs'], weights):
            self.backend.update_priority(idx, priority)


class PrioritizedReplayRollerEpsGreedyFactory(EnvRollerFactory):
    """ Factory class for PrioritizedReplayQRoller """

    def __init__(self, epsilon_schedule: Schedule, buffer_capacity: int, buffer_initial_size: int,
                 frame_stack: int, priority_exponent: float, priority_weight: Schedule, priority_epsilon: float):
        self.epsilon_schedule = epsilon_schedule
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack = frame_stack
        self.priority_exponent = priority_exponent
        self.priority_weight = priority_weight
        self.priority_epsilon = priority_epsilon

    def instantiate(self, environment, device, settings):
        return PrioritizedReplayRollerEpsGreedy(
            epsilon_schedule=self.epsilon_schedule,
            environment=environment,
            device=device,
            batch_size=settings.batch_size,
            buffer_capacity=self.buffer_capacity,
            buffer_initial_size=self.buffer_initial_size,
            frame_stack=self.frame_stack,
            priority_exponent=self.priority_exponent,
            priority_weight=self.priority_weight,
            priority_epsilon=self.priority_epsilon
        )


def create(epsilon_schedule: Schedule, buffer_capacity: int, buffer_initial_size: int, frame_stack: int,
           priority_exponent: float, priority_weight: Schedule, priority_epsilon: float):
    return PrioritizedReplayRollerEpsGreedyFactory(
        epsilon_schedule=epsilon_schedule,
        buffer_capacity=buffer_capacity,
        buffer_initial_size=buffer_initial_size,
        frame_stack=frame_stack,
        priority_exponent=priority_exponent,
        priority_weight=priority_weight,
        priority_epsilon=priority_epsilon
    )
