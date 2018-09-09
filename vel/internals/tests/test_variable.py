import yaml
import nose.tools as t

import vel.internals.variable as v


def test_variable_parsing():
    v.Variable.register()

    yaml_text = """
x:
  y: !var xxx 
"""

    yaml_contents = yaml.safe_load(yaml_text)

    t.assert_equal(yaml_contents['x']['y'].name, 'xxx')
