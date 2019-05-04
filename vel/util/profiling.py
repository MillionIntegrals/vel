import contextlib
import time


@contextlib.contextmanager
def timing_context(label=None):
    """ Measure time of expression as a context """
    start = time.time()
    yield
    end = time.time()

    if label is None:
        print("Context took {:.2f}s".format(end - start))
    else:
        print("{} took {:.2f}s".format(label, end - start))
