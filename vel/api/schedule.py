class Schedule:
    """ A schedule class encoding some kind of interpolation of a single value """
    def value(self, progress_indicator):
        """ Value at given progress step """
        raise NotImplementedError
