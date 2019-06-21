import torch

from vel.rl.api import Policy
from vel.util.tensor_util import to_device


class PolicyActor:
    """ Evaluates policy on a fixed set of environments. Additionally tracks the state """

    def __init__(self, num_envs: int, policy: Policy, device: torch.device):
        self.num_envs = num_envs
        self.policy = policy.to(device)
        self.device = device
        self.state = to_device(self.policy.zero_state(num_envs), self.device)

    def act(self, observation, advance_state=True, deterministic=False):
        """ Return result of a policy on a given input """
        result = self.policy.act(observation, state=self.state, deterministic=deterministic)

        if self.policy.is_stateful and advance_state:
            self.state = result['state']

        return result

    def reset_states(self, dones):
        """ Reset states given dones """
        if not self.policy.is_stateful:
            return

        dones = dones.to(self.device)

        self.state = self.policy.reset_state(self.state, dones)

    def value(self, observation):
        """ Return value for provided observations """
        return self.policy.value(observation, state=self.state)

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return self.policy.is_stateful
