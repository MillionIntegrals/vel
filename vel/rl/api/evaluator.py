class EvaluatorMeta(type):
    """ Metaclass for Evaluator - gathers all provider methods in a class attribute """
    def __new__(mcs, name, bases, attributes):
        providers = {}
        use_cache = {}

        for name, attr in attributes.items():
            if callable(attr):
                proper_name = getattr(attr, '_vel_evaluator_provides', None)
                if proper_name is not None:
                    providers[proper_name] = attr
                
                cache = getattr(attr, '_vel_use_cache', None)
                if cache is not None:
                    use_cache[proper_name] = cache

        attributes['_use_cache'] = use_cache
        attributes['_providers'] = providers

        return super().__new__(mcs, name, bases, attributes)


class Evaluator(metaclass=EvaluatorMeta):
    """
    Different models may have different outputs and approach evaluating environment differently.

    Evaluator is an object that abstracts over that, providing unified interface between algorithms
    which just need certain outputs from models and models that may provide them in different ways.

    I'll try to maintain here a dictionary of possible common values that can be requested from the evaluator.
    Rollouts should communicate using the same names

    - rollout:estimated_returns
        - Bootstrapped return (sum of discounted future rewards) estimated using returns and value estimates
    - rollout:values
        - Value estimates from the model that was used to generate the rollout
    - rollout:estimated_advantages
        - Advantage of a rollout (state, action) pair by the model that was used to generate the rollout
    - rollout:actions
        - Actions performed in a rollout
    - rollout:logprobs
        - Logarithm of probability for **all** actions of a policy used to perform rollout
        (defined only for finite action spaces)
    - rollout:action:logprobs
        - Logarithm of probability only for selected actions
    - rollout:dones
        - Whether given observation is last in a trajectory
    - rollout:dones
        - Raw rewards received from the environment in this learning process
    - rollout:final_values
        - Value estimates for observation after final observation in the rollout
    - rollout:observations
        - Observations of the rollout
    - rollout:observations_next
        - Next observations in the rollout
    - rollout:weights
        - Error weights of rollout samples
    - rollout:q
        - Action-values for each action in current space
        (defined only for finite action spaces)

    - model:logprobs
        - Logarithm of probability of **all** actions in an environment as in current model policy
        (defined only for finite action spaces)
    - model:q
        - Action-value for **all** actions
        (defined only for finite action spaces)
    - model:q_dist
        - Action-value histogram for **all** actions
        (defined only for finite action spaces)
    - model:q_dist_next
        - Action-value histogram for **all** actions from the 'next' state in the rollout
        (defined only for finite action spaces)
    - model:q_next
        - Action-value for **all** actions from the 'next' state in the rollout
        (defined only for finite action spaces)
    - model:entropy
        - Policy entropy for selected states
    - model:action:q
        - Action-value for actions selected in the rollout
    - model:model_action:q
        - Action-value for actions that model would perform (Deterministic policy only)
    - model:actions
        - Actions that model would perform (Deterministic policy only)
    - model:action:logprobs
        - Logarithm of probability for performed actions
    - model:policy_params
        - Parametrizations of policy for each state
    - model:values
        - Value estimates for each state, estimated by the current model
    - model:values_next
        - Value estimates for 'next' state of each transition
    """

    @staticmethod
    def provides(name, cache=True):
        """ Function decorator - value provided by the evaluator """
        def decorator(func):
            func._vel_evaluator_provides = name
            func._vel_use_cache = cache

            return func

        return decorator

    def __init__(self, rollout):
        self._storage = {}
        self.rollout = rollout

    def is_provided(self, name):
        """ Capability check if evaluator provides given value """
        if name in self._storage:
            return True
        elif name in self._providers:
            return True
        elif name.startswith('rollout:'):
            rollout_name = name[8:]
            return self.is_provided(rollout_name)
        else:
            return False

    def get(self, name, cache=True):
        """
        Return a value from this evaluator.

        Because tensor calculated is cached, it may lead to suble bugs if the same value is used multiple times
        with and without no_grad() context.

        It is advised in such cases to not use no_grad and stick to .detach()

        If you want to disable the cache you can pass 'cache=False' to the decorator to disable it  
        for the attribute or to the get() function to disable it just for that call
        """
        if name in self._use_cache and not self._use_cache[name]:
            cache = False
        
        if name in self._storage and cache:
            value = self._storage[name]
        elif name in self._providers:
            value = self._providers[name](self)
        elif name.startswith('rollout:'):
            rollout_name = name[8:]
            value = self.rollout.batch_tensor(rollout_name)
        else:
            raise RuntimeError(f"Key {name} is not provided by this evaluator")
       
        if cache:
            self._storage[name] = value
        
        return value

    def provide(self, name, value):
        """ Provide given value under specified name """
        self._storage[name] = value
