import threading
import contextlib


class Context(threading.local):
    """ Context object maintaining argument stack """
    def __init__(self):
        self.stack = []

    def push(self, **kwargs):
        self.stack.append(kwargs)

    def pop(self):
        self.stack.pop()

    def peek(self, name, default_value=None):
        index = len(self.stack) - 1

        while index >= 0:
            if name in self.stack[index]:
                return self.stack[index][name]

        return default_value


_THREAD_LOCAL_CONTEXT = Context()


@contextlib.contextmanager
def with_context(**kwargs):
    """ Add given variables to the context """
    _THREAD_LOCAL_CONTEXT.push(**kwargs)

    try:
        yield
    finally:
        _THREAD_LOCAL_CONTEXT.pop()


def push_context(**kwargs):
    """ Push custom context """
    _THREAD_LOCAL_CONTEXT.push(**kwargs)


def pop_context():
    """ Pop context from the stack """
    return _THREAD_LOCAL_CONTEXT.pop()


def get_context(name, default_value=None):
    """ Get context value (possibly default value) """
    return _THREAD_LOCAL_CONTEXT.peek(name, default_value)
