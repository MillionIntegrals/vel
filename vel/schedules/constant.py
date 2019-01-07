from vel.api import Schedule


class ConstantSchedule(Schedule):
    """ Interpolate variable linearly between start value and final value """

    def __init__(self, value):
        self._value = value

    def value(self, progress_indicator):
        """ Interpolate linearly between start and end """
        return self._value


def create(value):
    """ Vel factory function """
    return ConstantSchedule(value)
