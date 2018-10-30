import torch
import torch.optim

from vel.api import TrainingInfo, EpochInfo
from vel.rl.metrics import EpisodeRewardMetric
from vel.storage.streaming.stdout import StdoutStreaming
from vel.util.random import set_seed
from vel.rl.env.mujoco import MujocoEnv
from vel.rl.models.deterministic_policy_model import DeterministicPolicyModelFactory
from vel.rl.models.backbone.mlp import MLPFactory
from vel.rl.reinforcers.buffered_single_off_policy_iteration_reinforcer import (
    BufferedSingleOffPolicyIterationReinforcer, BufferedSingleOffPolicyIterationReinforcerSettings
)
from vel.rl.algo.policy_gradient.ddpg import DeepDeterministicPolicyGradient
from vel.rl.env_roller.single.deque_replay_roller_ou_noise import DequeReplayRollerOuNoise
from vel.optimizers.adam import AdamFactory


def half_cheetah_ddpg():
    device = torch.device('cuda:0')
    seed = 1002

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    env = MujocoEnv('HalfCheetah-v2').instantiate(seed=seed)

    model_factory = DeterministicPolicyModelFactory(
        policy_backbone=MLPFactory(input_length=17, hidden_layers=[64, 64], activation='tanh'),
        value_backbone=MLPFactory(input_length=23, hidden_layers=[64, 64], activation='tanh'),
    )

    model = model_factory.instantiate(action_space=env.action_space)

    reinforcer = BufferedSingleOffPolicyIterationReinforcer(
        device=device,
        settings=BufferedSingleOffPolicyIterationReinforcerSettings(
            batch_rollout_rounds=100,
            batch_training_rounds=50,
            batch_size=64,
            discount_factor=0.99
        ),
        environment=env,
        model=model,
        algo=DeepDeterministicPolicyGradient(
            model_factory=model_factory,
            tau=0.01,
        ),
        env_roller=DequeReplayRollerOuNoise(
            environment=env,
            device=device,
            batch_size=64,
            buffer_capacity=1_000_000,
            buffer_initial_size=2_000,
            noise_std_dev=0.2,
            normalize_observations=True,
            normalize_returns=True,
            discount_factor=0.99
        )
    )

    # Optimizer helper - A weird regularization settings I've copied from OpenAI code
    adam_optimizer = AdamFactory(
        lr=[1.0e-4, 1.0e-3, 1.0e-3],
        weight_decay=[0.0, 0.0, 0.001],
        eps=1.0e-4,
        layer_groups=True
    ).instantiate(model)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[StdoutStreaming()]  # Print live metrics every epoch to standard output
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 20 batches per epoch to average metrics nicely
    num_epochs = int(1.0e6 / 64 / 20)

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=20,
            optimizer=adam_optimizer
        )

        reinforcer.train_epoch(epoch_info)

    training_info.on_train_end()


if __name__ == '__main__':
    half_cheetah_ddpg()
