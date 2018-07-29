import torch
import numpy as np


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0

    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.-done)  # fixed off by one bug
        discounted.append(r)

    return discounted[::-1]


class StepEnvRoller:
    """ Class calculating env rollouts """

    def __init__(self, environment, device, number_of_steps, discount_factor):
        self.environment = environment
        self.device = device
        self.number_of_steps = number_of_steps
        self.discount_factor = discount_factor

        # Initial observation
        self.observation = self.environment.reset()
        self.dones = np.array([False for _ in range(self.observation.shape[0])])

        self.batch_observation_shape = (
            (self.observation.shape[0]*self.number_of_steps,) + self.environment.observation_space.shape
        )

    @torch.no_grad()
    def rollout(self, model):
        """ Calculate env rollout """

        observation_accumulator = []
        action_accumulator = []
        value_accumulator = []
        dones_accumulator = []
        rewards_accumulator = []
        episode_information = []

        # TODO(jerry): all this can be made numpy arrays to optimize accumulation
        # TODO(jerry): discounting can be done better/more efficient

        for step_idx in range(self.number_of_steps):
            model_input = torch.from_numpy(self.observation).to(self.device)
            actions, values, _ = model.step(model_input)

            actions = actions.detach().cpu().numpy()
            values = values.detach().cpu().numpy()

            observation_accumulator.append(self.observation)
            action_accumulator.append(actions)
            value_accumulator.append(values)
            dones_accumulator.append(self.dones)

            new_obs, new_rewards, new_dones, new_infos = self.environment.step(actions)

            # Done is flagged true when the episode has ended AND the frame we see is already a first frame from the
            # Next episode
            self.dones = new_dones
            self.observation = new_obs

            rewards_accumulator.append(new_rewards)

            for info in new_infos:
                maybe_episode_info = info.get('episode')

                if maybe_episode_info:
                    episode_information.append(maybe_episode_info)

        final_model_input = torch.from_numpy(self.observation).to(self.device)
        last_values = model.value(final_model_input)

        dones_accumulator.append(self.dones)

        # Swapaxes is important to make discounting process easier
        observation_buffer = np.asarray(observation_accumulator, dtype=np.uint8).swapaxes(0, 1)
        rewards_buffer = np.asarray(rewards_accumulator, dtype=np.float32).swapaxes(1, 0)
        # Action has no dtype, cause there may be different actions
        actions_buffer = np.asarray(action_accumulator).swapaxes(1, 0)
        values_buffer = np.asarray(value_accumulator, dtype=np.float32).swapaxes(1, 0)
        dones_buffer = np.asarray(dones_accumulator, dtype=np.bool).swapaxes(1, 0)

        masks_buffer = dones_buffer[:, :-1]
        dones_buffer = dones_buffer[:, 1:]

        discounted_rewards = self.discount_bootstrap(rewards_buffer, dones_buffer, last_values)

        # Reshape into final batch size
        return {
            'observations': observation_buffer.reshape(self.batch_observation_shape),
            'discounted_rewards': discounted_rewards.flatten(),
            'masks': masks_buffer.flatten(),
            'actions': actions_buffer.flatten(),
            'values': values_buffer.flatten(),
            'episode_information': episode_information
        }

    def discount_bootstrap(self, rewards_buffer, dones_buffer, values_buffer):
        true_value_buffer = np.zeros_like(rewards_buffer)

        # TODO(jerry) this probably can be turned into some numpy-cumsum
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(rewards_buffer, dones_buffer, values_buffer)):
            rewards = rewards.tolist()
            dones = dones.tolist()

            if dones[-1] == 0:
                # If the episode is not finished, add the last value
                rewards = discount_with_dones(rewards+[value], dones+[0], self.discount_factor)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.discount_factor)

            true_value_buffer[n] = rewards

        return true_value_buffer
