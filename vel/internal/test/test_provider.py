import os
import pytest

import vel.internal.provider as v
import vel.internal.parser as p
import vel.exception as e


def data_function(a, b):
    return a + b


def test_simple_instantiation():
    provider = v.Provider({
        'a': 1,
        'b': 2,
    })

    assert provider.instantiate_from_data(1) == 1
    assert provider.instantiate_from_data("abc") == "abc"
    assert provider.instantiate_from_data([1, 2, 3]) == [1, 2, 3]
    assert provider.instantiate_from_data({"a": "a", "b": "b"}) == {"a": "a", "b": "b"}


def test_instantiate_function_call():
    provider = v.Provider({
        'a': 1,
        'b': 2,
    })

    assert provider.resolve_and_call(data_function) == 3
    assert provider.resolve_and_call(data_function, extra_env={'b': 4}) == 5


def test_simple_injection():
    provider = v.Provider({
        'a': 1,
        'b': 2,
        'one': {
            'name': 'vel.internal.test.fixture_a'
        },

        'two': {
            'name': 'vel.internal.test.fixture_a',
            'a': 5,
            'b': 6
        },

        'three': {
            'name': 'vel.internal.test.fixture_b',
            'd': 'd'
        }
    })

    one = provider.instantiate_by_name('one')

    assert isinstance(one, dict)
    assert one['a'] == 1
    assert one['b'] == 2

    two = provider.instantiate_by_name('two')

    assert isinstance(two, dict)
    assert two['a'] == 5
    assert two['b'] == 6

    three = provider.instantiate_by_name('three')
    assert isinstance(three, dict)
    assert id(three['one']) == id(one)
    assert id(three['one']) != id(two)
    assert three['d'] == 'd'


def test_parameter_resolution():
    os.environ['TEST_VAR'] = '10'

    provider = v.Provider({
        'a': 1,
        'b': p.Parameter("xxx"),
        'one': {
            'name': 'vel.internal.test.fixture_a'
        },
        'two': {
            'name': 'vel.internal.test.fixture_a',
            'b': p.Parameter('yyy')
        },

        'three': {
            'name': 'vel.internal.test.fixture_a',
            'b': p.Parameter('yyy', 7)
        },

        'four': {
            'name': 'vel.internal.test.fixture_a',
            'b': p.EnvironmentVariable('TEST_VAR')
        },

    }, parameters={'xxx': 5})

    one = provider.instantiate_by_name('one')

    assert one['b'] == 5

    with pytest.raises(e.VelException):
        provider.instantiate_by_name('two')

    three = provider.instantiate_by_name('three')

    assert three['b'] == 7

    four = provider.instantiate_by_name('four')

    assert four['b'] == '10'


def test_render_configuration():
    os.environ['TEST_VAR'] = '10'

    provider = v.Provider({
        'a': 1,
        'b': p.Parameter("xxx"),
        'one': {
            'name': 'vel.internal.test.fixture_a'
        },
        'two': {
            'name': 'vel.internal.test.fixture_a',
            'b': p.Parameter('yyy', 5)
        },

        'three': {
            'name': 'vel.internal.test.fixture_a',
            'b': p.Parameter('yyy', 7)
        },

        'four': {
            'name': 'vel.internal.test.fixture_a',
            'b': p.EnvironmentVariable('TEST_VAR')
        },

    }, parameters={'xxx': 5})

    configuration = provider.render_configuration()

    assert configuration == {
        'a': 1,
        'b': 5,
        'one': {
            'name': 'vel.internal.test.fixture_a'
        },
        'two': {
            'name': 'vel.internal.test.fixture_a',
            'b': 5
        },

        'three': {
            'name': 'vel.internal.test.fixture_a',
            'b': 7
        },

        'four': {
            'name': 'vel.internal.test.fixture_a',
            'b': '10'
        },
    }
