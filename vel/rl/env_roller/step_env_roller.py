import torch
import numpy as np

from vel.api import BatchInfo
from vel.openai.baselines.common.vec_env import VecEnv
from vel.rl.api import Trajectories, Rollout, EnvRollerBase, EnvRollerFactoryBase, RlPolicy
from vel.rl.util.actor import PolicyActor
from vel.util.tensor_accumulator import TensorAccumulator


class StepEnvRoller(EnvRollerBase):
    """
    Class calculating env rollouts.
    """

    def __init__(self, environment: VecEnv, policy: RlPolicy, device: torch.device):
        self._environment = environment
        self.device = device

        # Initial observation - kept on CPU
        self.last_observation = torch.from_numpy(self.environment.reset()).clone()

        self.actor = PolicyActor(self.environment.num_envs, policy, device)

    @property
    def environment(self):
        """ Return environment of this env roller """
        return self._environment

    @torch.no_grad()
    def rollout(self, batch_info: BatchInfo, number_of_steps: int) -> Rollout:
        """ Calculate env rollout """
        accumulator = TensorAccumulator()
        episode_information = []  # List of dictionaries with episode information

        for step_idx in range(number_of_steps):
            step = self.actor.act(self.last_observation.to(self.device))

            # Add step to the tensor accumulator
            for name, tensor in step.items():
                # Take not that here we convert all the tensors to CPU
                accumulator.add(name, tensor.cpu())

            accumulator.add('observations', self.last_observation)

            actions_numpy = step['actions'].detach().cpu().numpy()
            new_obs, new_rewards, new_dones, new_infos = self.environment.step(actions_numpy)

            # Done is flagged true when the episode has ended AND the frame we see is already a first frame from the
            # next episode
            dones_tensor = torch.from_numpy(new_dones.astype(np.float32)).clone()

            self.last_observation = torch.from_numpy(new_obs).clone()
            self.actor.reset_states(dones_tensor)

            accumulator.add('dones', dones_tensor)
            accumulator.add('rewards', torch.from_numpy(new_rewards.astype(np.float32)).clone())

            episode_information.append(new_infos)

        final_values = self.actor.value(self.last_observation.to(self.device)).cpu()

        accumulated_tensors = accumulator.result()

        return Trajectories(
            num_steps=accumulated_tensors['observations'].size(0),
            num_envs=accumulated_tensors['observations'].size(1),
            environment_information=episode_information,
            transition_tensors=accumulated_tensors,
            rollout_tensors={
                'final_values': final_values
            }
        )


class StepEnvRollerFactory(EnvRollerFactoryBase):
    """ Factory for the StepEnvRoller """
    def __init__(self):
        pass

    def instantiate(self, environment, policy, device):
        return StepEnvRoller(
            environment=environment,
            policy=policy,
            device=device,
        )


def create():
    """ Vel factory function """
    return StepEnvRollerFactory()
