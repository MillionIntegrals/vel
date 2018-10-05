import torch
import numpy as np

from vel.openai.baselines.common.vec_env import VecEnv
from vel.rl.api.base import ReplayEnvRollerBase, EnvRollerFactory
from vel.rl.buffers.deque_multi_env_buffer_backend import DequeMultiEnvBufferBackend


class ReplayQEnvRoller(ReplayEnvRollerBase):
    """
    Class calculating env rollouts and storing them in a buffer for experience replay
    Idea behind this class is to store as much as we can as pytorch tensors to minimize tensor copying.
    """

    def __init__(self, environment: VecEnv, device, number_of_steps, discount_factor, buffer_capacity,
                 buffer_initial_size, frame_stack_compensation):
        self._environment = environment
        self.device = device
        self.number_of_steps = number_of_steps
        self.discount_factor = discount_factor
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack_compensation = frame_stack_compensation

        # Initial observation
        self.last_observation = self._to_tensor(self.environment.reset())
        self.dones = torch.tensor([False for _ in range(self.last_observation.shape[0])], device=self.device)

        self.batch_observation_shape = (
                (self.last_observation.shape[0] * self.number_of_steps,) + self.environment.observation_space.shape
        )

        # Replay buffer
        self.replay_buffer = DequeMultiEnvBufferBackend(
            buffer_capacity=self.buffer_capacity,
            num_envs=self.environment.num_envs,
            observation_space=self.environment.observation_space,
            action_space=self.environment.action_space,
            extra_data={
                'action_logits': np.zeros(
                    (self.buffer_capacity, self.environment.num_envs, self.environment.action_space.n), dtype=np.float32
                )
            },
            frame_stack_compensation=self.frame_stack_compensation is not None
        )

    @property
    def environment(self):
        """ Return environment of this env roller """
        return self._environment

    def _to_tensor(self, numpy_array):
        """ Convert numpy array to a tensor """
        return torch.from_numpy(numpy_array).to(self.device)

    @torch.no_grad()
    def rollout(self, batch_info, model):
        """ Calculate env rollout """
        observation_accumulator = []  # Device tensors
        action_accumulator = []  # Device tensors
        action_logit_accumulator = []  # Device tensors
        dones_accumulator = []  # Device tensors
        rewards_accumulator = []  # Device tensors
        episode_information = []  # Python objects

        for step_idx in range(self.number_of_steps):
            step = model.step(self.last_observation)

            actions = step['actions']

            observation_accumulator.append(self.last_observation)
            action_accumulator.append(actions)
            dones_accumulator.append(self.dones)

            action_logits = step['action_logits']
            action_logit_accumulator.append(action_logits)

            actions_numpy = actions.detach().cpu().numpy()
            new_obs, new_rewards, new_dones, new_infos = self.environment.step(actions_numpy)

            # Store rollout in the experience replay buffer
            self.replay_buffer.store_transition(
                frame=self.last_observation.detach().cpu().numpy(),
                action=actions_numpy,
                reward=new_rewards,
                done=new_dones,
                extra_info={
                    'action_logits': action_logits.detach().cpu().numpy(),
                }
            )

            # Done is flagged true when the episode has ended AND the frame we see is already a first frame from the
            # Next episode
            self.dones = self._to_tensor(new_dones.astype(np.uint8))
            self.last_observation = self._to_tensor(new_obs[:])

            rewards_accumulator.append(self._to_tensor(new_rewards.astype(np.float32)))

            for info in new_infos:
                maybe_episode_info = info.get('episode')

                if maybe_episode_info:
                    episode_information.append(maybe_episode_info)

        final_values = model.value(self.last_observation)

        dones_accumulator.append(self.dones)

        observation_buffer = torch.stack(observation_accumulator)
        rewards_buffer = torch.stack(rewards_accumulator)
        actions_buffer = torch.stack(action_accumulator)
        dones_buffer = torch.stack(dones_accumulator)
        action_logit_buffer = torch.stack(action_logit_accumulator)

        masks_buffer = dones_buffer[:-1, :]
        dones_buffer = dones_buffer[1:, :]

        batch_action_shape = (action_logit_buffer.size(0) * action_logit_buffer.size(1), action_logit_buffer.size(2))

        # Reshape into final batch size
        return {
            'size': self.batch_observation_shape[0],
            'observations': observation_buffer.reshape(self.batch_observation_shape),
            'masks': masks_buffer.flatten(),  # Dones and masks are basically the same, just shifted by 1
            'dones': dones_buffer.flatten(),
            'rewards': rewards_buffer.flatten(),
            'actions': actions_buffer.flatten(),
            'episode_information': episode_information,
            'action_logits': action_logit_buffer.reshape(batch_action_shape),
            'final_values': final_values
        }

    def is_ready_for_sampling(self) -> bool:
        """ If buffer is ready for drawing samples from it (usually checks if there is enough data) """
        return self.replay_buffer.current_size >= self.buffer_initial_size

    @torch.no_grad()
    def sample(self, batch_info, model):
        """ Sample experience from replay buffer and return a batch """
        rollout_idx = self.replay_buffer.sample_batch_rollout(
            rollout_length=self.number_of_steps, history_length=self.frame_stack_compensation
        )

        rollout = self.replay_buffer.get_rollout(
            rollout_idx, rollout_length=self.number_of_steps, history_length=self.frame_stack_compensation
        )

        action_logits_tensor = self._to_tensor(rollout['action_logits'])

        final_values = model.value(self._to_tensor(rollout['states+1'][-1]))

        return {
            'observations': self._to_tensor(rollout['states']).view(self.batch_observation_shape),
            'dones': self._to_tensor(rollout['dones'].astype(np.uint8)).flatten(),
            'rewards': self._to_tensor(rollout['rewards']).flatten(),
            'actions': self._to_tensor(rollout['actions']).flatten(),
            'action_logits': action_logits_tensor.view(
                action_logits_tensor.size(0) * action_logits_tensor.size(1), action_logits_tensor.size(2)
            ),
            'final_values': final_values
        }


class ReplayQEnvRollerFactory(EnvRollerFactory):
    """ Factory for the StepEnvRoller """
    def __init__(self, buffer_capacity, buffer_initial_size, frame_stack_compensation=None):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack_compensation = frame_stack_compensation

    def instantiate(self, environment, device, settings):
        return ReplayQEnvRoller(
            environment, device, settings.number_of_steps, settings.discount_factor,
            self.buffer_capacity, self.buffer_initial_size,
            frame_stack_compensation=self.frame_stack_compensation
        )


def create(buffer_capacity, buffer_initial_size, frame_stack_compensation=None):
    return ReplayQEnvRollerFactory(
        buffer_capacity=buffer_capacity,
        buffer_initial_size=buffer_initial_size,
        frame_stack_compensation=frame_stack_compensation
    )
