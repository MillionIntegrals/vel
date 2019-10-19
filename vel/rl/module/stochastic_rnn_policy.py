import gym

from vel.api import VModule, BackboneModule
from vel.rl.module.head.stochastic_action_head import make_stockastic_action_head
from vel.rl.module.head.value_head import ValueHead


class StochasticRnnPolicy(VModule):
    """
    Most generic policy gradient model class with a set of common actor-critic heads that share a single backbone
    RNN version
    """

    def __init__(self, net: BackboneModule, action_space: gym.Space):
        super().__init__()

        self.net = net

        assert self.net.is_stateful, "Must have a stateful backbone"

        (action_size, value_size) = self.net.size_hints().assert_tuple(2)

        self.action_head = make_stockastic_action_head(
            action_space=action_space,
            input_dim=action_size.last(),
        )
        self.value_head = ValueHead(
            input_dim=value_size.last()
        )

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return True

    def zero_state(self, batch_size):
        return self.net.zero_state(batch_size)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.net.reset_weights()
        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observations, state):
        """ Calculate model outputs """
        (action_hidden, value_hidden), new_state = self.net(observations, state=state)

        action_output = self.action_head(action_hidden)
        value_output = self.value_head(value_hidden)

        return action_output, value_output, new_state

    def reset_state(self, state, dones):
        """ Reset the state after the episode has been terminated """
        if (dones > 0).any().item():
            dones_expanded = dones.unsqueeze(-1)

            zero_state = self.net.zero_state(dones.shape[0])

            out_state = {}

            for key in state:
                state_item = state[key]
                zero_state_item = zero_state[key].to(state_item.device)

                final_item = state_item * (1 - dones_expanded) + zero_state_item * dones_expanded

                out_state[key] = final_item

            return out_state
        else:
            return state
