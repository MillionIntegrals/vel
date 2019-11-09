import torch
import numpy as np

from vel.api import BatchInfo
from vel.openai.baselines.common.vec_env import VecEnv
from vel.rl.api import Trajectories, Rollout, EnvRollerBase, EnvRollerFactoryBase, RlPolicy
from vel.rl.util.actor import PolicyActor
from vel.util.tensor_util import TensorAccumulator, to_device
from vel.util.datastructure import flatten_dict


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
        self.actor.eval()
        accumulator = TensorAccumulator()
        episode_information = []  # List of dictionaries with episode information

        for step_idx in range(number_of_steps):
            step = self.actor.act(self.last_observation.to(self.device), deterministic=False)
            cpu_step = to_device(step, torch.device('cpu'))

            # Add step to the tensor accumulator
            for name, tensor in cpu_step.items():

                # Take not that here we convert all the tensors to CPU
                accumulator.add(name, tensor)

            accumulator.add('observations', self.last_observation)

            actions_numpy = cpu_step['actions'].detach().numpy()
            new_obs, new_rewards, new_dones, new_infos = self.environment.step(actions_numpy)

            # Done is flagged true when the episode has ended AND the frame we see is already a first frame from the
            # next episode
            dones_tensor = torch.from_numpy(new_dones.astype(np.float32)).clone()

            self.last_observation = torch.from_numpy(new_obs).clone()
            self.actor.reset_states(dones_tensor)

            accumulator.add('dones', dones_tensor)
            accumulator.add('rewards', torch.from_numpy(new_rewards.astype(np.float32)).clone())

            episode_information.append(new_infos)

        accumulated_tensors = accumulator.result()

        # Perform last agent step, without advancing the state
        final_obs = self.actor.act(self.last_observation.to(self.device), advance_state=False)
        cpu_final_obs = to_device(final_obs, torch.device('cpu'))

        rollout_tensors = {}

        flatten_dict(cpu_final_obs, rollout_tensors, root='final')

        return Trajectories(
            num_steps=accumulated_tensors['observations'].size(0),
            num_envs=accumulated_tensors['observations'].size(1),
            environment_information=episode_information,
            transition_tensors=accumulated_tensors,
            rollout_tensors=rollout_tensors
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
