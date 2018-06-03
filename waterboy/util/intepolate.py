import numpy as np
import warnings


def interpolate_linear(start, end, steps):
    """ Interpolate series between start and end in given number of steps - linear interpolation """
    return np.linspace(start, end, steps)


def interpolate_logscale(start, end, steps):
    """ Interpolate series between start and end in given number of steps - logscale interpolation """
    if start <= 0.0:
        warnings.warn("Start of logscale interpolation must be positive!")
        start = 1e-5

    return np.logspace(np.log10(float(start)), np.log10(float(end)), steps)


INTERP_DICT = {
    'linear': interpolate_linear,
    'logscale': interpolate_logscale
}


def interpolate_series(start, end, steps, how='linear'):
    """ Interpolate series between start and end in given number of steps """
    return INTERP_DICT[how](start, end, steps)

