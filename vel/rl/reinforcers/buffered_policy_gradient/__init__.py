from .buffered_policy_gradient_reinforcer import BufferedPolicyGradientReinforcerFactory, BufferedPolicyGradientSettings


def create(model_config, model, vec_env, policy_gradient, env_roller,
           number_of_steps, parallel_envs, discount_factor, batch_size=256,
           experience_replay=1, stochastic_experience_replay=True):
    """ Create a policy gradient reinforcer - factory """
    settings = BufferedPolicyGradientSettings(
        number_of_steps=number_of_steps,
        discount_factor=discount_factor,
        batch_size=batch_size,
        experience_replay=experience_replay,
        stochastic_experience_replay=stochastic_experience_replay
    )

    return BufferedPolicyGradientReinforcerFactory(
        settings,
        env_factory=vec_env,
        model_factory=model,
        parallel_envs=parallel_envs,
        env_roller_factory=env_roller,
        policy_gradient=policy_gradient,
        seed=model_config.seed
    )
