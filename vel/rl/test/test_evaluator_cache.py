from vel.rl.api import Evaluator, Rollout

calls = {
    "a": 0,
    "b": 0,
    "c": 0,
}

class TestEvaluator(Evaluator):
    @Evaluator.provides('test:a')
    def test_a(self):
        calls["a"] += 1
    
    @Evaluator.provides('test:b', cache=False)
    def test_b(self):
        calls["b"] += 1
    
    @Evaluator.provides('test:c')
    def test_c(self):
        calls["c"] += 1
    

def test_evaluator():
    e = TestEvaluator(Rollout())
    e.get("test:a")
    e.get("test:a")
    e.get("test:a")
    
    e.get("test:b")
    e.get("test:b")
    e.get("test:b")

    e.get("test:c")
    e.get("test:c")
    e.get("test:c", cache=False)

    assert calls["a"] == 1 # test:a is cached so just one call
    assert calls["b"] == 3 # test:b is never cached so three calls
    assert calls["c"] == 2 # test:c is cached but one call is not so two calls