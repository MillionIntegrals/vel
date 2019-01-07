import vel.util.intepolate as interpolate

from vel.api import Schedule


class LinearAndConstantSchedule(Schedule):
    """
    Interpolate variable linearly between start value and final values
    only up to given point and return constant value afterwards.
    """

    def __init__(self, initial_value, final_value, end_of_interpolation):
        self.initial_value = initial_value
        self.final_value = final_value
        self.end_of_interpolation = end_of_interpolation

    def value(self, progress_indicator):
        """ Interpolate linearly between start and end """
        if progress_indicator <= self.end_of_interpolation:
            return interpolate.interpolate_linear_single(
                self.initial_value, self.final_value, progress_indicator/self.end_of_interpolation
            )
        else:
            return self.final_value


def create(initial_value, final_value, end_of_interpolation):
    """ Vel factory function """
    return LinearAndConstantSchedule(initial_value, final_value, end_of_interpolation)
