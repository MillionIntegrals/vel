import typing


def flatten_dict(dictionary: dict, output: typing.Optional[dict] = None, root: str = '') -> dict:
    """ From a nested dictionary built a flat version, concatenating keys with '.' """
    if output is None:
        output = {}

    for key, value in dictionary.items():
        if isinstance(value, dict):
            if root:
                flatten_dict(value, output, f"{root}.{key}")
            else:
                flatten_dict(value, output, key)
        else:
            if root:
                output[f"{root}.{key}"] = value
            else:
                output[key] = value

    return output
