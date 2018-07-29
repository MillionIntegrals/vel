from .policy_gradient_reinforcer import PolicyGradientReinforcerFactory


def create(policy_gradient, vec_env, number_of_steps, parallel_envs, discount_factor, seed, max_grad_norm=None):
    """ Create a policy gradient reinforcer - factory """
    return PolicyGradientReinforcerFactory(
        policy_gradient=policy_gradient,
        vec_env=vec_env,
        number_of_steps=number_of_steps,
        parallel_envs=parallel_envs,
        discount_factor=discount_factor,
        seed=seed,
        max_grad_norm=max_grad_norm
    )
