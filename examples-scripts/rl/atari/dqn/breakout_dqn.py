import torch
import torch.optim as optim

from vel.rl.metrics import EpisodeRewardMetric
from vel.storage.streaming.stdout import StdoutStreaming
from vel.util.random import set_seed

from vel.rl.env.classic_atari import ClassicAtariEnv
from vel.rl.vecenv.subproc import SubprocVecEnvWrapper

from vel.rl.models.q_model import QModelFactory
from vel.rl.models.backbone.nature_cnn import NatureCnnFactory

from vel.rl.reinforcers.buffered_single_off_policy_iteration_reinforcer import (
    BufferedSingleOffPolicyIterationReinforcer, 
    BufferedSingleOffPolicyIterationReinforcerSettings 
)

from vel.schedules.linear_and_constant import LinearAndConstantSchedule
from vel.rl.algo.dqn import DeepQLearning 
from vel.rl.env_roller.single.deque_replay_roller_epsgreedy import DequeReplayRollerEpsGreedy 

from vel.api.info import TrainingInfo, EpochInfo
from vel.rl.commands.rl_train_command import FrameTracker


def breakout_dqn():
    device = torch.device('cuda:0')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    # Create 16 environments evaluated in parallel in sub processess with all usual DeepMind wrappers
    # These are just helper functions for that
    vec_env = SubprocVecEnvWrapper(
        ClassicAtariEnv('BreakoutNoFrameskip-v4')
    ).instantiate(parallel_envs=8, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model = QModelFactory(
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    )

    # Reinforcer - an object managing the learning process
    reinforcer = BufferedSingleOffPolicyIterationReinforcer(
        device=device,
        settings=BufferedSingleOffPolicyIterationReinforcerSettings(
            batch_training_rounds=4,
            batch_rollout_rounds=4,
            batch_size=32,
            discount_factor=0.99
        ),
        model=model.instantiate(action_space=vec_env.action_space),
        environment=vec_env,
        algo=DeepQLearning(
            model_factory=model,
            target_update_frequency=10000,
            double_dqn=False,
            max_grad_norm=0.5,
        ),
        env_roller=DequeReplayRollerEpsGreedy(
            environment=vec_env,
            device=device,
            buffer_capacity=250000,
            buffer_initial_size=30000,
            frame_stack=4,
            batch_size=32,
            epsilon_schedule=LinearAndConstantSchedule(
                end_of_interpolation=0.1,
                initial_value=1.0,
                final_value=0.1,
            )
        )
    )

    # Model optimizer
    optimizer = optim.RMSprop(reinforcer.model.parameters(), lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1.0e-1)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[
            StdoutStreaming(),   # Print live metrics every epoch to standard output
            FrameTracker(1.1e7)      # We need frame tracker to track the progress of learning
        ]
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 10 batches per epoch to average metrics nicely
    # Rollout size is 8 environments times 128 steps
    num_epochs = int(1.1e7 / (128 * 8) / 10)

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=2500,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info)

    training_info.on_train_end()


if __name__ == '__main__':
    breakout_dqn()
