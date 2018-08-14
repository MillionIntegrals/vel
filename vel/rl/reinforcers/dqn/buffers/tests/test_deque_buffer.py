import nose
import numpy as np

from vel.exceptions import VelException
from ..deque_buffer import DequeBufferBackend


def test_simple_get_frame():
    buffer = DequeBufferBackend(20, (2, 2, 1))

    v1 = np.ones(4).reshape((2, 2, 1))
    v2 = v1 * 2
    v3 = v1 * 3

    buffer.store_transition(v1, 0, 0, False)
    buffer.store_transition(v2, 0, 0, False)
    buffer.store_transition(v3, 0, 0, False)

    assert np.all(buffer.get_frame(0, 4).max(0).max(0) == np.array([0, 0, 0, 1]))
    assert np.all(buffer.get_frame(1, 4).max(0).max(0) == np.array([0, 0, 1, 2]))
    assert np.all(buffer.get_frame(2, 4).max(0).max(0) == np.array([0, 1, 2, 3]))

    with nose.tools.assert_raises(VelException):
        buffer.get_frame(3, 4)

