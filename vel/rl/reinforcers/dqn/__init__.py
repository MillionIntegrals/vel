from .dqn_reinforcer import DqnReinforcerFactory, DqnReinforcerSettings


def create(model_config, model, env, buffer, epsilon_schedule, train_frequency: int, batch_size: int, target_update_frequency: int,
           discount_factor: float, max_grad_norm: float, double_dqn: bool = False):
    """ Vel creation function for DqnReinforcerFactory """
    settings = DqnReinforcerSettings(
        buffer=buffer,
        epsilon_schedule=epsilon_schedule,
        train_frequency=train_frequency,
        batch_size=batch_size,
        double_dqn=double_dqn,
        target_update_frequency=target_update_frequency,
        discount_factor=discount_factor,
        max_grad_norm=max_grad_norm
    )

    return DqnReinforcerFactory(settings, env_factory=env, model_factory=model, seed=model_config.seed)
