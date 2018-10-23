import torch
import numpy as np

from vel.openai.baselines.common.vec_env import VecEnv
from vel.rl.api import Rollout, Trajectories
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
        self.last_observation_cpu = self.environment.reset()
        self.last_observation = self._to_tensor(self.last_observation_cpu)

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
    def rollout(self, batch_info, model) -> Rollout:
        """ Calculate env rollout """
        observation_accumulator = []  # Device tensors
        action_accumulator = []  # Device tensors
        logprob_accumulator = []  # Device tensors
        done_accumulator = []  # Device tensors
        reward_accumulator = []  # Device tensors
        episode_information = []  # Python objects

        for step_idx in range(self.number_of_steps):
            step = model.step(self.last_observation)
            actions = step['actions']

            observation_accumulator.append(self.last_observation)
            action_accumulator.append(actions)

            logprobs = step['logprobs']
            logprob_accumulator.append(logprobs)

            actions_numpy = actions.detach().cpu().numpy()
            new_obs, new_rewards, new_dones, new_infos = self.environment.step(actions_numpy)

            # Store rollout in the experience replay buffer
            self.replay_buffer.store_transition(
                frame=self.last_observation_cpu,
                action=actions_numpy,
                reward=new_rewards,
                done=new_dones,
                extra_info={
                    'action_logits': logprobs.detach().cpu().numpy(),
                }
            )

            # Done is flagged true when the episode has ended AND the frame we see is already a first frame from the
            # Next episode
            self.last_observation_cpu = new_obs[:]
            self.last_observation = self._to_tensor(self.last_observation_cpu)

            done_accumulator.append(self._to_tensor(new_dones.astype(np.float32)))
            reward_accumulator.append(self._to_tensor(new_rewards.astype(np.float32)))

            episode_information.append(new_infos)

        final_values = model.value(self.last_observation)

        observations_buffer = torch.stack(observation_accumulator)
        rewards_buffer = torch.stack(reward_accumulator)
        actions_buffer = torch.stack(action_accumulator)
        dones_buffer = torch.stack(done_accumulator)
        action_logit_buffer = torch.stack(logprob_accumulator)

        return Trajectories(
            num_steps=self.number_of_steps,
            num_envs=self.environment.num_envs,
            environment_information=episode_information,
            transition_tensors={
                'observations': observations_buffer,
                'rewards': rewards_buffer,
                'dones': dones_buffer,
                'actions': actions_buffer,
                'logprobs': action_logit_buffer,
            },
            rollout_tensors={
                'final_estimated_values': final_values
            }
        )

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

        return Trajectories(
            num_steps=self.number_of_steps,
            num_envs=self.environment.num_envs,
            environment_information=None,
            transition_tensors={
                'observations': self._to_tensor(rollout['states']),
                'dones': self._to_tensor(rollout['dones'].astype(np.float32)),
                'rewards': self._to_tensor(rollout['rewards']),
                'actions': self._to_tensor(rollout['actions']),
                'logprobs': self._to_tensor(rollout['action_logits'])
            },
            rollout_tensors={
                'final_estimated_values': model.value(self._to_tensor(rollout['states+1'][-1]))
            }
        )


class ReplayQEnvRollerFactory(EnvRollerFactory):
    """ Factory for the StepEnvRoller """
    def __init__(self, buffer_capacity, buffer_initial_size, number_of_steps, frame_stack_compensation=None):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack_compensation = frame_stack_compensation
        self.number_of_steps = number_of_steps

    def instantiate(self, environment, device, settings):
        return ReplayQEnvRoller(
            environment, device, self.number_of_steps, settings.discount_factor,
            self.buffer_capacity, self.buffer_initial_size,
            frame_stack_compensation=self.frame_stack_compensation
        )


def create(buffer_capacity, buffer_initial_size, number_of_steps, frame_stack_compensation=None):
    return ReplayQEnvRollerFactory(
        buffer_capacity=buffer_capacity,
        number_of_steps=number_of_steps,
        buffer_initial_size=buffer_initial_size,
        frame_stack_compensation=frame_stack_compensation
    )
