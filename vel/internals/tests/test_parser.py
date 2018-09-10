import nose.tools as t

import vel.internals.parser as v


def setup_parser():
    """ Set up test environment """
    v.Parser.register()


@t.with_setup(setup_parser)
def test_variable_parsing():
    v.Parser.register()

    yaml_text = """
x:
  y: !param xxx 
"""

    yaml_contents = v.Parser.parse(yaml_text)

    t.assert_is_instance(yaml_contents['x']['y'], v.Parameter)
    t.assert_equal(yaml_contents['x']['y'].name, 'xxx')


@t.with_setup(setup_parser)
def test_env_variable_parsing():
    v.Parser.register()

    yaml_text = """
x:
  y: !env xxx 
"""

    yaml_contents = v.Parser.parse(yaml_text)

    t.assert_is_instance(yaml_contents['x']['y'], v.EnvironmentVariable)
    t.assert_equal(yaml_contents['x']['y'].name, 'xxx')


@t.with_setup(setup_parser)
def test_variable_default_values():
    v.Parser.register()

    yaml_text = """
x:
  y: !param xxx = 5
"""

    yaml_contents = v.Parser.parse(yaml_text)

    t.assert_equal(yaml_contents['x']['y'].default_value, 5)

    yaml_text = """
x:
  y: !param xxx = abc
"""

    yaml_contents = v.Parser.parse(yaml_text)

    t.assert_equal(yaml_contents['x']['y'].default_value, 'abc')

    yaml_text = """
x:
  y: !param xxx = 'abc def'
"""

    yaml_contents = v.Parser.parse(yaml_text)

    t.assert_equal(yaml_contents['x']['y'].default_value, 'abc def')

    yaml_text = """
x:
  y: !param xxx = null
"""

    yaml_contents = v.Parser.parse(yaml_text)

    t.assert_equal(yaml_contents['x']['y'].default_value, None)

    yaml_text = """
x:
  y: !param xxx
"""

    yaml_contents = v.Parser.parse(yaml_text)

    t.assert_equal(yaml_contents['x']['y'].default_value, v.DUMMY_VALUE)


def test_parse_equality():
    t.assert_equal(v.Parser.parse_equality("x=5"), ('x', 5))
    t.assert_equal(v.Parser.parse_equality("  x   =   5  "), ('x', 5))

    with t.assert_raises(AssertionError):
        v.Parser.parse_equality("  1   =   2  ")

    t.assert_equal(v.Parser.parse_equality("  'asd'   =   'www zzz'  "), ('asd', 'www zzz'))

    t.assert_equal(v.Parser.parse_equality("  'asd'   =   'www=zzz'  "), ('asd', 'www=zzz'))
