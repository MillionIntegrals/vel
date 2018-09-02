from .policy_gradient_reinforcer import PolicyGradientReinforcerFactory, PolicyGradientSettings


def create(model_config, model, vec_env, policy_gradient, env_roller, number_of_steps, parallel_envs,
           discount_factor, max_grad_norm=None, gae_lambda=1.0, batch_size=256, experience_replay=1):
    """ Create a policy gradient reinforcer - factory """
    settings = PolicyGradientSettings(
        number_of_steps=number_of_steps,
        discount_factor=discount_factor,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        experience_replay=experience_replay
    )

    return PolicyGradientReinforcerFactory(
        settings,
        env_factory=vec_env,
        model_factory=model,
        policy_gradient=policy_gradient,
        parallel_envs=parallel_envs,
        env_roller_factory=env_roller,
        seed=model_config.seed
    )
