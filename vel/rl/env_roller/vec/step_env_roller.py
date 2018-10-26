import torch
import numpy as np

from vel.rl.api.base import EnvRollerBase, EnvRollerFactory
from vel.rl.api import Trajectories


class StepEnvRoller(EnvRollerBase):
    """
    Class calculating env rollouts.
    Idea behind this class is to store as much as we can as pytorch tensors to minimize tensor copying.
    """

    def __init__(self, environment, device, number_of_steps, discount_factor, gae_lambda=1.0):
        self._environment = environment
        self.device = device
        self.number_of_steps = number_of_steps
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda

        # Initial observation
        self.last_observation = self._to_tensor(self.environment.reset())

        # Relevant for RNN policies
        self.hidden_state = None

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
        value_accumulator = []  # Device tensors
        dones_accumulator = []  # Device tensors
        rewards_accumulator = []  # Device tensors
        episode_information = []  # Python objects
        logprobs_accumulator = []  # Device tensors

        if self.hidden_state is None and model.is_recurrent:
            self.hidden_state = torch.zeros(
                (self.last_observation.size(0), model.state_dim),
                device=self.device,
                dtype=torch.float32
            )

        # Remember rollout initial state, we'll use that for learning as well
        initial_hidden_state = self.hidden_state

        for step_idx in range(self.number_of_steps):
            if model.is_recurrent:
                step = model.step(self.last_observation, state=self.hidden_state)
                self.hidden_state = step['state']
            else:
                step = model.step(self.last_observation)

            actions, values, logprobs = step['actions'], step['values'], step['logprobs']

            observation_accumulator.append(self.last_observation)
            action_accumulator.append(actions)
            value_accumulator.append(values)
            logprobs_accumulator.append(logprobs)

            actions_numpy = actions.detach().cpu().numpy()
            new_obs, new_rewards, new_dones, new_infos = self.environment.step(actions_numpy)

            # Done is flagged true when the episode has ended AND the frame we see is already a first frame from the
            # next episode

            dones_tensor = self._to_tensor(new_dones.astype(np.float32))
            dones_accumulator.append(dones_tensor)

            self.last_observation = self._to_tensor(new_obs[:])

            if model.is_recurrent:
                # Zero out state in environments that have finished
                self.hidden_state = self.hidden_state * (1.0 - dones_tensor.unsqueeze(-1))

            rewards_accumulator.append(self._to_tensor(new_rewards.astype(np.float32)))

            episode_information.append(new_infos)

        if model.is_recurrent:
            final_values = model.value(self.last_observation, state=self.hidden_state)
        else:
            final_values = model.value(self.last_observation)

        observations_buffer = torch.stack(observation_accumulator)
        rewards_buffer = torch.stack(rewards_accumulator)
        actions_buffer = torch.stack(action_accumulator)  # Actions may have various different dtypes
        values_buffer = torch.stack(value_accumulator)
        dones_buffer = torch.stack(dones_accumulator)
        logprobs_buffer = torch.stack(logprobs_accumulator)

        # Generalized Advantage Estimation
        # https://arxiv.org/abs/1506.02438
        advantages = self.discount_bootstrap_gae(
            rewards_buffer, dones_buffer, values_buffer, final_values,
            self.discount_factor, self.gae_lambda
        )

        returns = advantages + values_buffer

        return Trajectories(
            num_steps=advantages.size(0),
            num_envs=advantages.size(1),
            environment_information=episode_information,
            transition_tensors={
                'observations': observations_buffer,
                'estimated_returns': returns,
                'dones': dones_buffer,
                'actions': actions_buffer,
                'estimated_values': values_buffer,
                'estimated_advantages': advantages,
                'action:logprobs': logprobs_buffer,
            },
            rollout_tensors={
                'initial_hidden_state': initial_hidden_state,
                'final_estimated_values': final_values
            }
        )

    def discount_bootstrap(self, rewards_buffer, dones_buffer, final_values, discount_factor):
        """ Calculate state values bootstrapping off the following state values """
        true_value_buffer = torch.zeros_like(rewards_buffer)
        dones_buffer = dones_buffer

        # discount/bootstrap off value fn
        current_value = final_values

        for i in reversed(range(self.number_of_steps)):
            current_value = rewards_buffer[i] + discount_factor * current_value * (1.0 - dones_buffer[i])
            true_value_buffer[i] = current_value

        return true_value_buffer

    def discount_bootstrap_gae(self, rewards_buffer, dones_buffer, final_values, last_values_buffer,
                               discount_factor, gae_lambda):
        """ Calculate state values bootstrapping off the following state values - Generalized Advantage Estimation """
        advantage_buffer = torch.zeros_like(rewards_buffer)
        advantage_buffer = advantage_buffer
        dones_buffer = dones_buffer

        # Accmulate sums
        sum_accumulator = 0

        for i in reversed(range(self.number_of_steps)):
            if i == self.number_of_steps - 1:
                next_value = last_values_buffer
            else:
                next_value = final_values[i + 1]

            bellman_delta = (
                    rewards_buffer[i] + discount_factor * next_value * (1.0 - dones_buffer[i]) - final_values[i]
            )

            advantage_buffer[i] = sum_accumulator = (
                bellman_delta + discount_factor * gae_lambda * sum_accumulator * (1.0 - dones_buffer[i])
            )

        return advantage_buffer


class StepEnvRollerFactory(EnvRollerFactory):
    """ Factory for the StepEnvRoller """
    def __init__(self, number_of_steps, gae_lambda=1.0):
        self.gae_lambda = gae_lambda
        self.number_of_steps = number_of_steps

    def instantiate(self, environment, device, settings):
        return StepEnvRoller(
            environment=environment,
            device=device,
            number_of_steps=self.number_of_steps,
            discount_factor=settings.discount_factor,
            gae_lambda=self.gae_lambda
        )


def create(number_of_steps, gae_lambda=1.0):
    return StepEnvRollerFactory(number_of_steps=number_of_steps, gae_lambda=gae_lambda)
