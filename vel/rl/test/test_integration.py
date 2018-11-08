import torch
import torch.optim as optim

from vel.rl.commands.rl_train_command import FrameTracker
from vel.rl.metrics import EpisodeRewardMetric
from vel.schedules.linear import LinearSchedule
from vel.schedules.linear_and_constant import LinearAndConstantSchedule
from vel.util.random import set_seed

from vel.rl.env.classic_atari import ClassicAtariEnv
from vel.rl.env.mujoco import MujocoEnv
from vel.rl.vecenv.subproc import SubprocVecEnvWrapper
from vel.rl.vecenv.dummy import DummyVecEnvWrapper

from vel.rl.models.policy_gradient_model import PolicyGradientModelFactory
from vel.rl.models.q_policy_gradient_model import QPolicyGradientModelFactory
from vel.rl.models.q_model import QModelFactory
from vel.rl.models.deterministic_policy_model import DeterministicPolicyModelFactory
from vel.rl.models.policy_gradient_model_separate import PolicyGradientModelSeparateFactory

from vel.rl.models.backbone.nature_cnn import NatureCnnFactory
from vel.rl.models.backbone.mlp import MLPFactory

from vel.rl.reinforcers.on_policy_iteration_reinforcer import (
    OnPolicyIterationReinforcer, OnPolicyIterationReinforcerSettings
)

from vel.rl.reinforcers.buffered_single_off_policy_iteration_reinforcer import (
    BufferedSingleOffPolicyIterationReinforcer, BufferedSingleOffPolicyIterationReinforcerSettings
)

from vel.rl.reinforcers.buffered_mixed_policy_iteration_reinforcer import (
    BufferedMixedPolicyIterationReinforcer, BufferedMixedPolicyIterationReinforcerSettings
)

from vel.rl.algo.dqn import DeepQLearning
from vel.rl.algo.policy_gradient.a2c import A2CPolicyGradient
from vel.rl.algo.policy_gradient.ppo import PpoPolicyGradient
from vel.rl.algo.policy_gradient.trpo import TrpoPolicyGradient
from vel.rl.algo.policy_gradient.acer import AcerPolicyGradient
from vel.rl.algo.policy_gradient.ddpg import DeepDeterministicPolicyGradient

from vel.rl.env_roller.vec.step_env_roller import StepEnvRoller
from vel.rl.env_roller.vec.replay_q_env_roller import ReplayQEnvRoller
from vel.rl.env_roller.single.deque_replay_roller_epsgreedy import DequeReplayRollerEpsGreedy
from vel.rl.env_roller.single.prioritized_replay_roller_epsgreedy import PrioritizedReplayRollerEpsGreedy
from vel.rl.env_roller.single.deque_replay_roller_ou_noise import DequeReplayRollerOuNoise

from vel.api.info import TrainingInfo, EpochInfo


def test_a2c_breakout():
    """
    Simple 1 iteration of a2c breakout
    """
    device = torch.device('cpu')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    # Create 16 environments evaluated in parallel in sub processess with all usual DeepMind wrappers
    # These are just helper functions for that
    vec_env = SubprocVecEnvWrapper(
        ClassicAtariEnv('BreakoutNoFrameskip-v4'), frame_history=4
    ).instantiate(parallel_envs=16, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model = PolicyGradientModelFactory(
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    ).instantiate(action_space=vec_env.action_space)

    # Reinforcer - an object managing the learning process
    reinforcer = OnPolicyIterationReinforcer(
        device=device,
        settings=OnPolicyIterationReinforcerSettings(
            discount_factor=0.99,
            batch_size=256,
        ),
        model=model,
        algo=A2CPolicyGradient(
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            max_grad_norm=0.5
        ),
        env_roller=StepEnvRoller(
            environment=vec_env,
            device=device,
            number_of_steps=5,
            discount_factor=0.99,
        )
    )

    # Model optimizer
    optimizer = optim.RMSprop(reinforcer.model.parameters(), lr=7.0e-4, eps=1e-3)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[]  # Print live metrics every epoch to standard output
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 100 batches per epoch to average metrics nicely
    num_epochs = 1

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=1,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info, interactive=False)

    training_info.on_train_end()


def test_ppo_breakout():
    """
    Simple 1 iteration of ppo breakout
    """
    device = torch.device('cpu')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    # Create 16 environments evaluated in parallel in sub processess with all usual DeepMind wrappers
    # These are just helper functions for that
    vec_env = SubprocVecEnvWrapper(
        ClassicAtariEnv('BreakoutNoFrameskip-v4'), frame_history=4
    ).instantiate(parallel_envs=8, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model = PolicyGradientModelFactory(
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    ).instantiate(action_space=vec_env.action_space)

    # Reinforcer - an object managing the learning process
    reinforcer = OnPolicyIterationReinforcer(
        device=device,
        settings=OnPolicyIterationReinforcerSettings(
            discount_factor=0.99,
            batch_size=4,
            experience_replay=2,
        ),
        model=model,
        algo=PpoPolicyGradient(
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            max_grad_norm=0.5,
            cliprange=LinearSchedule(0.1, 0.0),
            normalize_advantage=True
        ),
        env_roller=StepEnvRoller(
            environment=vec_env,
            device=device,
            number_of_steps=12,
            discount_factor=0.99,
        )
    )

    # Model optimizer
    # optimizer = optim.RMSprop(reinforcer.model.parameters(), lr=7.0e-4, eps=1e-3)
    optimizer = optim.Adam(reinforcer.model.parameters(), lr=2.5e-4, eps=1e-5)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[
            FrameTracker(100_000)
        ]  # Print live metrics every epoch to standard output
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 100 batches per epoch to average metrics nicely
    num_epochs = 1

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=1,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info, interactive=False)

    training_info.on_train_end()


def test_dqn_breakout():
    """
    Simple 1 iteration of DQN breakout
    """
    device = torch.device('cpu')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    # Only single environment for DQN
    env = ClassicAtariEnv('BreakoutNoFrameskip-v4').instantiate(seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model_factory = QModelFactory(
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    )

    # Reinforcer - an object managing the learning process
    reinforcer = BufferedSingleOffPolicyIterationReinforcer(
        device=device,
        settings=BufferedSingleOffPolicyIterationReinforcerSettings(
            batch_rollout_rounds=4,
            batch_training_rounds=1,
            batch_size=32,
            discount_factor=0.99
        ),
        environment=env,
        algo=DeepQLearning(
            model_factory=model_factory,
            double_dqn=False,
            target_update_frequency=10_000,
            max_grad_norm=0.5
        ),
        model=model_factory.instantiate(action_space=env.action_space),
        env_roller=DequeReplayRollerEpsGreedy(
            environment=env,
            device=device,
            epsilon_schedule=LinearAndConstantSchedule(initial_value=1.0, final_value=0.1, end_of_interpolation=0.1),
            batch_size=32,
            buffer_capacity=100,
            buffer_initial_size=100,
            frame_stack=4
        )
    )

    # Model optimizer
    optimizer = optim.RMSprop(reinforcer.model.parameters(), lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-3)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[
            FrameTracker(100_000)
        ]  # Print live metrics every epoch to standard output
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 100 batches per epoch to average metrics nicely
    num_epochs = 1

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=1,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info, interactive=False)

    training_info.on_train_end()


def test_prioritized_dqn_breakout():
    """
    Simple 1 iteration of DQN prioritized replay breakout
    """
    device = torch.device('cpu')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    # Only single environment for DQN
    env = ClassicAtariEnv('BreakoutNoFrameskip-v4').instantiate(seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model_factory = QModelFactory(
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    )

    # Reinforcer - an object managing the learning process
    reinforcer = BufferedSingleOffPolicyIterationReinforcer(
        device=device,
        settings=BufferedSingleOffPolicyIterationReinforcerSettings(
            batch_rollout_rounds=4,
            batch_training_rounds=1,
            batch_size=32,
            discount_factor=0.99
        ),
        environment=env,
        algo=DeepQLearning(
            model_factory=model_factory,
            double_dqn=False,
            target_update_frequency=10_000,
            max_grad_norm=0.5
        ),
        model=model_factory.instantiate(action_space=env.action_space),
        env_roller=PrioritizedReplayRollerEpsGreedy(
            environment=env,
            device=device,
            epsilon_schedule=LinearAndConstantSchedule(initial_value=1.0, final_value=0.1, end_of_interpolation=0.1),
            batch_size=8,
            buffer_capacity=100,
            priority_epsilon=1.0e-6,
            buffer_initial_size=100,
            frame_stack=4,
            priority_exponent=0.6,
            priority_weight=LinearSchedule(
                initial_value=0.4,
                final_value=1.0
            ),
        ),
    )

    # Model optimizer
    optimizer = optim.RMSprop(reinforcer.model.parameters(), lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-3)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[
            FrameTracker(100_000)
        ]  # Print live metrics every epoch to standard output
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 100 batches per epoch to average metrics nicely
    num_epochs = 1

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=1,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info, interactive=False)

    training_info.on_train_end()


def test_ddpg_bipedal_walker():
    """
    1 iteration of DDPG bipedal walker environment
    """
    device = torch.device('cpu')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    # Only single environment for DDPG
    env = MujocoEnv('BipedalWalker-v2').instantiate(seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model_factory = DeterministicPolicyModelFactory(
        policy_backbone=MLPFactory(input_length=24, hidden_layers=[64, 64], normalization='layer'),
        value_backbone=MLPFactory(input_length=28, hidden_layers=[64, 64], normalization='layer')
    )

    # Reinforcer - an object managing the learning process
    reinforcer = BufferedSingleOffPolicyIterationReinforcer(
        device=device,
        settings=BufferedSingleOffPolicyIterationReinforcerSettings(
            batch_rollout_rounds=4,
            batch_training_rounds=1,
            batch_size=32,
            discount_factor=0.99
        ),
        environment=env,
        algo=DeepDeterministicPolicyGradient(
            model_factory=model_factory,
            tau=0.01,
            max_grad_norm=0.5
        ),
        model=model_factory.instantiate(action_space=env.action_space),
        env_roller=DequeReplayRollerOuNoise(
            environment=env,
            device=device,
            batch_size=32,
            buffer_capacity=100,
            buffer_initial_size=100,
            noise_std_dev=0.2,
            normalize_observations=True,
            discount_factor=0.99
        )
    )

    # Model optimizer
    optimizer = optim.Adam(reinforcer.model.parameters(), lr=2.5e-4, eps=1e-4)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[
            FrameTracker(100_000)
        ]  # Print live metrics every epoch to standard output
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 100 batches per epoch to average metrics nicely
    num_epochs = 1

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=1,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info, interactive=False)

    training_info.on_train_end()


def test_trpo_bipedal_walker():
    """
    1 iteration of TRPO on bipedal walker
    """
    device = torch.device('cpu')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    vec_env = DummyVecEnvWrapper(
        MujocoEnv('BipedalWalker-v2'), normalize=True
    ).instantiate(parallel_envs=8, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model_factory = PolicyGradientModelSeparateFactory(
        policy_backbone=MLPFactory(input_length=24, hidden_layers=[32, 32]),
        value_backbone=MLPFactory(input_length=24, hidden_layers=[32])
    )

    # Reinforcer - an object managing the learning process
    reinforcer = OnPolicyIterationReinforcer(
        device=device,
        settings=OnPolicyIterationReinforcerSettings(
            discount_factor=0.99,
        ),
        model=model_factory.instantiate(action_space=vec_env.action_space),
        algo=TrpoPolicyGradient(
            max_kl=0.01,
            cg_iters=10,
            line_search_iters=10,
            improvement_acceptance_ratio=0.1,
            cg_damping=0.1,
            vf_iters=5,
            entropy_coef=0.0,
            max_grad_norm=0.5,
        ),
        env_roller=StepEnvRoller(
            environment=vec_env,
            device=device,
            number_of_steps=12,
            discount_factor=0.99,
        )
    )

    # Model optimizer
    optimizer = optim.Adam(reinforcer.model.parameters(), lr=1.0e-3, eps=1e-4)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[
            FrameTracker(100_000)
        ]  # Print live metrics every epoch to standard output
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 100 batches per epoch to average metrics nicely
    num_epochs = 1

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=1,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info, interactive=False)

    training_info.on_train_end()


def test_acer_breakout():
    """
    1 iteration of ACER on breakout environment
    """
    device = torch.device('cpu')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    # Create 16 environments evaluated in parallel in sub processess with all usual DeepMind wrappers
    # These are just helper functions for that
    vec_env = SubprocVecEnvWrapper(
        ClassicAtariEnv('BreakoutNoFrameskip-v4'), frame_history=4
    ).instantiate(parallel_envs=16, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model_factory = QPolicyGradientModelFactory(
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    )

    # Reinforcer - an object managing the learning process
    reinforcer = BufferedMixedPolicyIterationReinforcer(
        device=device,
        settings=BufferedMixedPolicyIterationReinforcerSettings(
            discount_factor=0.99,
            experience_replay=2,
            stochastic_experience_replay=False
        ),
        model=model_factory.instantiate(action_space=vec_env.action_space),
        env=vec_env,
        algo=AcerPolicyGradient(
            model_factory=model_factory,
            entropy_coefficient=0.01,
            q_coefficient=0.5,
            rho_cap=10.0,
            retrace_rho_cap=1.0,
            trust_region=True,
            trust_region_delta=1.0,
            max_grad_norm=10.0,
        ),
        env_roller=ReplayQEnvRoller(
            environment=vec_env,
            device=device,
            number_of_steps=12,
            discount_factor=0.99,
            buffer_capacity=100,
            buffer_initial_size=100,
            frame_stack_compensation=4
        ),
    )

    # Model optimizer
    optimizer = optim.RMSprop(reinforcer.model.parameters(), lr=7.0e-4, eps=1e-3, alpha=0.99)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[]  # Print live metrics every epoch to standard output
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 100 batches per epoch to average metrics nicely
    num_epochs = 1

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=1,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info, interactive=False)

    training_info.on_train_end()


