import typing


def process_environment_settings(default_dictionary: dict, settings: typing.Optional[dict]=None,
                                 presets: typing.Optional[dict]=None):
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

