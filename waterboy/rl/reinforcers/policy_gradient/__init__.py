from .policy_gradient_reinforcer import PolicyGradientReinforcerFactory, PolicyGradientSettings
from .policy_gradient_model import PolicyGradientModelAugmentor


def create(policy_gradient, vec_env, number_of_steps, parallel_envs, discount_factor, seed, max_grad_norm=None,
           model_augmentors=None, gae_lambda=1.0, batch_size=256, experience_replay=1):
    """ Create a policy gradient reinforcer - factory """
    if model_augmentors is None:
        model_augmentors = [PolicyGradientModelAugmentor()]

    settings = PolicyGradientSettings(
        policy_gradient=policy_gradient,
        vec_env=vec_env,
        model_augmentors=model_augmentors,
        parallel_envs=parallel_envs,
        number_of_steps=number_of_steps,
        discount_factor=discount_factor,
        gae_lambda=gae_lambda,
        seed=seed,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        experience_replay=experience_replay
    )

    return PolicyGradientReinforcerFactory(settings)
