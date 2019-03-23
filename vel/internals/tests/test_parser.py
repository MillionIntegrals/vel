import pytest

import vel.internals.parser as v


@pytest.fixture
def setup_parser():
    """ Set up test environment """
    v.Parser.register()


def test_variable_parsing(setup_parser):
    yaml_text = """
x:
  y: !param xxx 
"""

    yaml_contents = v.Parser.parse(yaml_text)

    assert isinstance(yaml_contents['x']['y'], v.Parameter)
    assert yaml_contents['x']['y'].name == 'xxx'


def test_env_variable_parsing(setup_parser):
    yaml_text = """
x:
  y: !env xxx 
"""

    yaml_contents = v.Parser.parse(yaml_text)

    assert isinstance(yaml_contents['x']['y'], v.EnvironmentVariable)
    assert yaml_contents['x']['y'].name == 'xxx'


def test_variable_default_values(setup_parser):
    yaml_text = """
x:
  y: !param xxx = 5
"""

    yaml_contents = v.Parser.parse(yaml_text)

    assert yaml_contents['x']['y'].default_value == 5

    yaml_text = """
x:
  y: !param xxx = abc
"""

    yaml_contents = v.Parser.parse(yaml_text)

    assert yaml_contents['x']['y'].default_value == 'abc'

    yaml_text = """
x:
  y: !param xxx = 'abc def'
"""

    yaml_contents = v.Parser.parse(yaml_text)

    assert yaml_contents['x']['y'].default_value == 'abc def'

    yaml_text = """
x:
  y: !param xxx = null
"""

    yaml_contents = v.Parser.parse(yaml_text)

    assert yaml_contents['x']['y'].default_value == None

    yaml_text = """
x:
  y: !param xxx
"""

    yaml_contents = v.Parser.parse(yaml_text)

    assert yaml_contents['x']['y'].default_value == v.DUMMY_VALUE


def test_parse_equality():
    assert v.Parser.parse_equality("x=5") == ('x', 5)
    assert v.Parser.parse_equality("  x   =   5  ") == ('x', 5)

    with pytest.raises(AssertionError):
        v.Parser.parse_equality("  1   =   2  ")

    assert v.Parser.parse_equality("  'asd'   =   'www zzz'  ") == ('asd', 'www zzz')
    assert v.Parser.parse_equality("  'asd'   =   'www=zzz'  ") == ('asd', 'www=zzz')
