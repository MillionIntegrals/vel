import gym
import typing

from vel.api import SizeHints, SizeHint


def process_environment_settings(default_dictionary: dict, settings: typing.Optional[dict] = None,
                                 presets: typing.Optional[dict] = None):
    """ Process a dictionary of env settings """
    settings = settings if settings is not None else {}
    presets = presets if presets is not None else {}

    env_keys = sorted(set(default_dictionary.keys()) | set(presets.keys()))

    result_dict = {}

    for key in env_keys:
        if key in default_dictionary:
            new_dict = default_dictionary[key].copy()
        else:
            new_dict = {}

        new_dict.update(settings)

        if key in presets:
            new_dict.update(presets[key])

        result_dict[key] = new_dict

    return result_dict


def gym_space_to_size_hint(space: gym.Space) -> SizeHints:
    """ Convert Gym observation space to size hints """
    if isinstance(space, gym.spaces.Box):
        return size_hint_from_shape(space.shape)
    else:
        raise NotImplementedError


def size_hint_from_shape(shape: typing.Tuple[int]) -> SizeHints:
    """ Convert tensor shape (without batch dimension) into a size hint """
    return SizeHints(SizeHint(*([None] + list(shape))))
