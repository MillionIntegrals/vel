import numpy as np

import vel.util.math as math_util
import vel.util.tensor_util as tensor_util


class Rollout:
    """ Base class for environment rollout data """

    def to_transitions(self) -> 'Transitions':
        """ Convert given rollout to Transitions """
        raise NotImplementedError

    def episode_information(self):
        """ List of information about finished episodes """
        raise NotImplementedError

    def frames(self) -> int:
        """ Number of frames in rollout """
        raise NotImplementedError

    def batch_tensor(self, name):
        """ A buffer of a given value in a 'flat' (minibatch-indexed) format """
        raise NotImplementedError

    def has_tensor(self, name):
        """ Return true if rollout contains tensor with given name """
        raise NotImplementedError

    def to_device(self, device):
        """ Move a rollout to a selected device """
        raise NotImplementedError


class Transitions(Rollout):
    """
    Rollout of random transitions, that don't necessarily have to be in any order

    transition_tensors - tensors that have a row (multidimensional) per each transition. E.g. state, reward, done
    """
    def __init__(self, size, environment_information, transition_tensors, extra_data=None):
        self.size = size
        self.environment_information = environment_information
        self.transition_tensors = transition_tensors
        self.extra_data = extra_data if extra_data is not None else {}

    def to_transitions(self) -> 'Transitions':
        """ Convert given rollout to Transitions """
        return self

    def episode_information(self):
        """ List of information about finished episodes """
        return [info.get('episode') for info in self.environment_information if 'episode' in info]

    def frames(self):
        """ Number of frames in this rollout """
        return self.size

    def shuffled_batches(self, batch_size):
        """ Generate randomized batches of data """
        if batch_size >= self.size:
            yield self
        else:
            batch_splits = math_util.divide_ceiling(self.size, batch_size)
            indices = list(range(self.size))
            np.random.shuffle(indices)

            for sub_indices in np.array_split(indices, batch_splits):
                yield Transitions(
                    size=len(sub_indices),
                    environment_information=None,
                    # Dont use it in batches for a moment, can be uncommented later if needed
                    # environment_information=[info[sub_indices.tolist()] for info in self.environment_information]
                    transition_tensors={k: v[sub_indices] for k, v in self.transition_tensors.items()}
                    # extra_data does not go into batches
                )

    def has_tensor(self, name):
        """ Return true if rollout contains tensor with given name """
        return name in self.transition_tensors

    def batch_tensor(self, name):
        """ A buffer of a given value in a 'flat' (minibatch-indexed) format """
        return self.transition_tensors[name]

    def to_device(self, device, non_blocking=True):
        """ Move a rollout to a selected device """
        return Transitions(
            size=self.size,
            environment_information=self.environment_information,
            transition_tensors={k: v.to(device, non_blocking=non_blocking) for k, v in self.transition_tensors.items()},
            extra_data=self.extra_data
        )


class Trajectories(Rollout):
    """
    Rollout of trajectories - a number of consecutive transitions

    transition_tensors - tensors that have a row (multidimensional) per each transition. E.g. state, reward, done
    rollout_tensors - tensors that have a row (multidimensional) per whole rollout. E.g. final_value, initial rnn state
    """
    def __init__(self, num_steps, num_envs, environment_information, transition_tensors, rollout_tensors, extra_data=None):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.environment_information = environment_information
        self.transition_tensors = transition_tensors
        self.rollout_tensors = rollout_tensors
        self.extra_data = extra_data if extra_data is not None else {}

    def to_transitions(self) -> 'Transitions':
        """ Convert given rollout to Transitions """
        # No need to propagate 'rollout_tensors' as they won't mean anything
        return Transitions(
            size=self.num_steps * self.num_envs,
            environment_information=
                [ei for l in self.environment_information for ei in l]
                if self.environment_information is not None else None,
            transition_tensors={
                name: tensor_util.merge_first_two_dims(t) for name, t in self.transition_tensors.items()
            },
            extra_data=self.extra_data
        )

    def shuffled_batches(self, batch_size):
        """ Generate randomized batches of data - only sample whole trajectories """
        if batch_size >= self.num_envs * self.num_steps:
            yield self
        else:
            rollouts_in_batch = batch_size // self.num_steps

            batch_splits = math_util.divide_ceiling(self.num_envs, rollouts_in_batch)

            indices = list(range(self.num_envs))
            np.random.shuffle(indices)

            for sub_indices in np.array_split(indices, batch_splits):
                yield Trajectories(
                    num_steps=self.num_steps,
                    num_envs=len(sub_indices),
                    # Dont use it in batches for a moment, can be uncommented later if needed
                    # environment_information=[x[sub_indices.tolist()] for x in self.environment_information],
                    environment_information=None,
                    transition_tensors={k: x[:, sub_indices] for k, x in self.transition_tensors.items()},
                    rollout_tensors={k: x[sub_indices] for k, x in self.rollout_tensors.items()},
                    # extra_data does not go into batches
                )

    def batch_tensor(self, name):
        """ A buffer of a given value in a 'flat' (minibatch-indexed) format """
        if name in self.transition_tensors:
            return tensor_util.merge_first_two_dims(self.transition_tensors[name])
        else:
            return self.rollout_tensors[name]

    def has_tensor(self, name):
        """ Return true if rollout contains tensor with given name """
        return (name in self.transition_tensors) or (name in self.rollout_tensors)

    def flatten_tensor(self, tensor):
        """ Merge first two dims of a tensor """
        return tensor_util.merge_first_two_dims(tensor)

    def episode_information(self):
        """ List of information about finished episodes """
        return [
            info.get('episode') for infolist in self.environment_information for info in infolist if 'episode' in info
        ]

    def frames(self):
        """ Number of frames in rollout """
        return self.num_steps * self.num_envs

    def to_device(self, device, non_blocking=True):
        """ Move a rollout to a selected device """
        return Trajectories(
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            environment_information=self.environment_information,
            transition_tensors={k: v.to(device, non_blocking=non_blocking) for k, v in self.transition_tensors.items()},
            rollout_tensors={k: v.to(device, non_blocking=non_blocking) for k, v in self.rollout_tensors.items()},
            extra_data=self.extra_data
        )
