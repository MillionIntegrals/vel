import nose.tools as t
import os

import vel.internals.provider as v
import vel.internals.parser as p
import vel.exceptions as e


def data_function(a, b):
    return a + b


def test_simple_instantiation():
    provider = v.Provider({
        'a': 1,
        'b': 2,
    })

    t.assert_equal(provider.instantiate_from_data(1), 1)
    t.assert_equal(provider.instantiate_from_data("abc"), "abc")
    t.assert_equal(provider.instantiate_from_data([1, 2, 3]), [1, 2, 3])
    t.assert_equal(provider.instantiate_from_data({"a": "a", "b": "b"}), {"a": "a", "b": "b"})


def test_instantiate_function_call():
    provider = v.Provider({
        'a': 1,
        'b': 2,
    })

    t.assert_equal(provider.resolve_and_call(data_function), 3)
    t.assert_equal(provider.resolve_and_call(data_function, extra_env={'b': 4}), 5)


def test_simple_injection():
    provider = v.Provider({
        'a': 1,
        'b': 2,
        'one': {
            'name': 'vel.internals.tests.fixture_a'
        },

        'two': {
            'name': 'vel.internals.tests.fixture_a',
            'a': 5,
            'b': 6
        },

        'three': {
            'name': 'vel.internals.tests.fixture_b',
            'd': 'd'
        }
    })

    one = provider.instantiate_by_name('one')

    t.assert_is_instance(one, dict)
    t.assert_equal(one['a'], 1)
    t.assert_equal(one['b'], 2)

    two = provider.instantiate_by_name('two')

    t.assert_is_instance(two, dict)
    t.assert_equal(two['a'], 5)
    t.assert_equal(two['b'], 6)

    three = provider.instantiate_by_name('three')
    t.assert_is_instance(three, dict)
    t.assert_equal(id(three['one']), id(one))
    t.assert_not_equal(id(three['one']), id(two))
    t.assert_equal(three['d'], 'd')


def test_parameter_resolution():
    os.environ['TEST_VAR'] = '10'

    provider = v.Provider({
        'a': 1,
        'b': p.Parameter("xxx"),
        'one': {
            'name': 'vel.internals.tests.fixture_a'
        },
        'two': {
            'name': 'vel.internals.tests.fixture_a',
            'b': p.Parameter('yyy')
        },

        'three': {
            'name': 'vel.internals.tests.fixture_a',
            'b': p.Parameter('yyy', 7)
        },

        'four': {
            'name': 'vel.internals.tests.fixture_a',
            'b': p.EnvironmentVariable('TEST_VAR')
        },

    }, parameters={'xxx': 5})

    one = provider.instantiate_by_name('one')

    t.assert_equal(one['b'], 5)

    with t.assert_raises(e.VelException):
        provider.instantiate_by_name('two')

    three = provider.instantiate_by_name('three')

    t.assert_equal(three['b'], 7)

    four = provider.instantiate_by_name('four')

    t.assert_equal(four['b'], '10')


def test_render_configuration():
    os.environ['TEST_VAR'] = '10'

    provider = v.Provider({
        'a': 1,
        'b': p.Parameter("xxx"),
        'one': {
            'name': 'vel.internals.tests.fixture_a'
        },
        'two': {
            'name': 'vel.internals.tests.fixture_a',
            'b': p.Parameter('yyy', 5)
        },

        'three': {
            'name': 'vel.internals.tests.fixture_a',
            'b': p.Parameter('yyy', 7)
        },

        'four': {
            'name': 'vel.internals.tests.fixture_a',
            'b': p.EnvironmentVariable('TEST_VAR')
        },

    }, parameters={'xxx': 5})

    configuration = provider.render_configuration()

    t.assert_equal(configuration, {
        'a': 1,
        'b': 5,
        'one': {
            'name': 'vel.internals.tests.fixture_a'
        },
        'two': {
            'name': 'vel.internals.tests.fixture_a',
            'b': 5
        },

        'three': {
            'name': 'vel.internals.tests.fixture_a',
            'b': 7
        },

        'four': {
            'name': 'vel.internals.tests.fixture_a',
            'b': '10'
        },
    })
