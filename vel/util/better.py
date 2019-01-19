import numpy as np


def better(old_value, new_value, mode):
    """ Check if new value is better than the old value"""
    if (old_value is None or np.isnan(old_value)) and (new_value is not None and not np.isnan(new_value)):
        return True

    if mode == 'min':
        return new_value < old_value
    elif mode == 'max':
        return new_value > old_value
    else:
        raise RuntimeError(f"Mode '{mode}' value is not supported")
