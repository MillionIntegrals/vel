import typing
import collections.abc as abc

from vel.exception import VelException


class SizeHint(tuple):
    """ Neural network hint of a layer size. Should consist of either integers or Nones """

    def __new__(cls, *args):
        return super().__new__(cls, tuple(args))

    def last(self) -> int:
        """ Return last part of the size hint, make sure it's not None """
        assert self[-1] is not None, "Size hint shouldn't be None"
        return self[-1]

    def shape(self, idx=1) -> typing.Tuple[int]:
        """ Get shape of size hint, except for a number of dimensions (batch dimensions """
        return self[idx:]

    def __repr__(self):
        internal = ", ".join([self._inner_repr(s) for s in self])
        return f"{self.__class__.__name__}({internal})"

    def _inner_repr(self, x):
        if x is None:
            return '-'
        else:
            return repr(x)


SizeTuple = typing.Tuple[SizeHint]
SizeDict = typing.Dict[str, SizeHint]


class SizeHints:
    """ SizeHint, tuple of size hints or dict of size hints """

    TYPE_NONE = 0
    TYPE_SIZE = 1
    TYPE_TUPLE = 2
    TYPE_DICT = 3

    def __init__(self, size_hints: typing.Union[SizeHint, SizeTuple, SizeDict] = None):
        self.size_hints = size_hints

        if self.size_hints is None:
            self.type = self.TYPE_NONE
        elif isinstance(self.size_hints, SizeHint):
            self.type = self.TYPE_SIZE
        elif isinstance(self.size_hints, abc.Sequence):
            self.size_hints = tuple(self.size_hints)
            self.type = self.TYPE_TUPLE
        elif isinstance(self.size_hints, abc.Mapping):
            self.type = self.TYPE_DICT
        else:
            raise VelException("Invalid size hints: {}".format(self.size_hints))

    def assert_tuple(self, length : typing.Optional[int] = None) -> SizeTuple:
        """ Assert given size hints is a tuple """
        assert self.type == self.TYPE_TUPLE, "Network needs to return a tuple"

        if length is not None:
            assert len(self.size_hints) == length, "Network must return {} results".format(length)

        return self.size_hints

    def assert_single(self, length: typing.Optional[int] = None) -> SizeHint:
        """ Make sure there is a single tensor as a size hint """
        assert self.type == self.TYPE_SIZE, "Layer input must be single tensor"

        if length is not None:
            assert len(self.size_hints) == length, f"Layer input must have shape [{length}]"

        return self.size_hints

    def unwrap(self):
        """ Return the underlying data """
        return self.size_hints

    def __repr__(self):
        return repr(self.size_hints)
