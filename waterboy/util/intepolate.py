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


def interpolate_cosine_single(start, end, coefficient):
    """ Cosine interpolation """
    cos_out = (np.cos(np.pi*coefficient) + 1) / 2.0
    return end + (start - end) * cos_out


def interpolate_linear_single(start, end, coefficient):
    """ Cosine interpolation """
    return start + (end - start) * coefficient


def interpolate_logscale_single(start, end, coefficient):
    """ Cosine interpolation """
    return np.exp(np.log(start) + (np.log(end) - np.log(start)) * coefficient)


INTERP_DICT = {
    'linear': interpolate_linear,
    'logscale': interpolate_logscale
}


INTERP_SINGLE_DICT = {
    'linear': interpolate_linear_single,
    'logscale': interpolate_logscale_single,
    'cosine': interpolate_cosine_single
}


def interpolate_series(start, end, steps, how='linear'):
    """ Interpolate series between start and end in given number of steps """
    return INTERP_DICT[how](start, end, steps)


def interpolate_single(start, end, coefficient, how='linear'):
    """ Interpolate single value between start and end in given number of steps """
    return INTERP_SINGLE_DICT[how](start, end, coefficient)

