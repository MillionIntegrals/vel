import vel.util.intepolate as interpolate

from vel.api import Schedule


class LinearSchedule(Schedule):
    """ Interpolate variable linearly between start value and final value """

    def __init__(self, initial_value, final_value):
        self.initial_value = initial_value
        self.final_value = final_value

    def value(self, progress_indicator):
        """ Interpolate linearly between start and end """
        return interpolate.interpolate_linear_single(self.initial_value, self.final_value, progress_indicator)


def create(initial_value, final_value):
    """ Vel factory function """
    return LinearSchedule(initial_value, final_value)

