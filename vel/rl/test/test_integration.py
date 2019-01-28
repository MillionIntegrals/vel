import torch
import torch.optim as optim

from vel.modules.input.image_to_tensor import ImageToTensorFactory
from vel.modules.input.normalize_observations import NormalizeObservationsFactory
from vel.rl.buffers.circular_replay_buffer import CircularReplayBuffer
from vel.rl.buffers.prioritized_circular_replay_buffer import PrioritizedCircularReplayBuffer
from vel.rl.commands.rl_train_command import FrameTracker
from vel.rl.env_roller.step_env_roller import StepEnvRoller
from vel.rl.env_roller.trajectory_replay_env_roller import TrajectoryReplayEnvRoller
from vel.rl.env_roller.transition_replay_env_roller import TransitionReplayEnvRoller
from vel.rl.metrics import EpisodeRewardMetric
from vel.rl.modules.noise.eps_greedy import EpsGreedy
from vel.rl.modules.noise.ou_noise import OuNoise
from vel.schedules.linear import LinearSchedule
from vel.schedules.linear_and_constant import LinearAndConstantSchedule
from vel.util.random import set_seed

from vel.rl.env.classic_atari import ClassicAtariEnv
from vel.rl.env.mujoco import MujocoEnv
from vel.rl.vecenv.subproc import SubprocVecEnvWrapper
from vel.rl.vecenv.dummy import DummyVecEnvWrapper

from vel.rl.models.stochastic_policy_model import StochasticPolicyModelFactory
from vel.rl.models.q_stochastic_policy_model import QStochasticPolicyModelFactory
from vel.rl.models.q_model import QModelFactory
from vel.rl.models.deterministic_policy_model import DeterministicPolicyModelFactory
from vel.rl.models.stochastic_policy_model_separate import StochasticPolicyModelSeparateFactory

from vel.rl.models.backbone.nature_cnn import NatureCnnFactory
from vel.rl.models.backbone.mlp import MLPFactory

from vel.rl.reinforcers.on_policy_iteration_reinforcer import (
    OnPolicyIterationReinforcer, OnPolicyIterationReinforcerSettings
)

from vel.rl.reinforcers.buffered_off_policy_iteration_reinforcer import (
    BufferedOffPolicyIterationReinforcer, BufferedOffPolicyIterationReinforcerSettings
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

from vel.api.info import TrainingInfo, EpochInfo


CPU_DEVICE = torch.device('cpu')


def test_a2c_breakout():
    """
    Simple 1 iteration of a2c breakout
    """
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
    model = StochasticPolicyModelFactory(
        input_block=ImageToTensorFactory(),
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    ).instantiate(action_space=vec_env.action_space)

    # Reinforcer - an object managing the learning process
    reinforcer = OnPolicyIterationReinforcer(
        device=CPU_DEVICE,
        settings=OnPolicyIterationReinforcerSettings(
            batch_size=256,
            number_of_steps=5
        ),
        model=model,
        algo=A2CPolicyGradient(
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            discount_factor=0.99,
            max_grad_norm=0.5
        ),
        env_roller=StepEnvRoller(
            environment=vec_env,
            device=CPU_DEVICE
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
    model = StochasticPolicyModelFactory(
        input_block=ImageToTensorFactory(),
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    ).instantiate(action_space=vec_env.action_space)

    # Reinforcer - an object managing the learning process
    reinforcer = OnPolicyIterationReinforcer(
        device=device,
        settings=OnPolicyIterationReinforcerSettings(
            number_of_steps=12,
            batch_size=4,
            experience_replay=2,
        ),
        model=model,
        algo=PpoPolicyGradient(
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            max_grad_norm=0.5,
            cliprange=LinearSchedule(0.1, 0.0),
            discount_factor=0.99,
            normalize_advantage=True
        ),
        env_roller=StepEnvRoller(
            environment=vec_env,
            device=device,
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
    vec_env = DummyVecEnvWrapper(
        ClassicAtariEnv('BreakoutNoFrameskip-v4'), frame_history=4
    ).instantiate(parallel_envs=1, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model_factory = QModelFactory(
        input_block=ImageToTensorFactory(),
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    )

    # Reinforcer - an object managing the learning process
    reinforcer = BufferedOffPolicyIterationReinforcer(
        device=device,
        settings=BufferedOffPolicyIterationReinforcerSettings(
            rollout_steps=4,
            training_steps=1,
        ),
        environment=vec_env,
        algo=DeepQLearning(
            model_factory=model_factory,
            double_dqn=False,
            target_update_frequency=10_000,
            discount_factor=0.99,
            max_grad_norm=0.5
        ),
        model=model_factory.instantiate(action_space=vec_env.action_space),
        env_roller=TransitionReplayEnvRoller(
            environment=vec_env,
            device=device,
            replay_buffer=CircularReplayBuffer(
                buffer_capacity=100,
                buffer_initial_size=100,
                num_envs=vec_env.num_envs,
                observation_space=vec_env.observation_space,
                action_space=vec_env.action_space,
                frame_stack_compensation=True,
                frame_history=4
            ),
            action_noise=EpsGreedy(
                epsilon=LinearAndConstantSchedule(
                    initial_value=1.0, final_value=0.1, end_of_interpolation=0.1
                ),
                environment=vec_env
            )
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
    vec_env = DummyVecEnvWrapper(
        ClassicAtariEnv('BreakoutNoFrameskip-v4'), frame_history=4
    ).instantiate(parallel_envs=1, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model_factory = QModelFactory(
        input_block=ImageToTensorFactory(),
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    )

    # Reinforcer - an object managing the learning process
    reinforcer = BufferedOffPolicyIterationReinforcer(
        device=device,
        settings=BufferedOffPolicyIterationReinforcerSettings(
            rollout_steps=4,
            training_steps=1,
        ),
        environment=vec_env,
        algo=DeepQLearning(
            model_factory=model_factory,
            double_dqn=False,
            target_update_frequency=10_000,
            discount_factor=0.99,
            max_grad_norm=0.5
        ),
        model=model_factory.instantiate(action_space=vec_env.action_space),
        env_roller=TransitionReplayEnvRoller(
            environment=vec_env,
            device=device,
            replay_buffer=PrioritizedCircularReplayBuffer(
                buffer_capacity=100,
                buffer_initial_size=100,
                num_envs=vec_env.num_envs,
                observation_space=vec_env.observation_space,
                action_space=vec_env.action_space,
                priority_exponent=0.6,
                priority_weight=LinearSchedule(
                    initial_value=0.4,
                    final_value=1.0
                ),
                priority_epsilon=1.0e-6,
                frame_stack_compensation=True,
                frame_history=4
            ),
            action_noise=EpsGreedy(
                epsilon=LinearAndConstantSchedule(
                    initial_value=1.0, final_value=0.1, end_of_interpolation=0.1
                ),
                environment=vec_env
            )
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


def test_ddpg_bipedal_walker():
    """
    1 iteration of DDPG bipedal walker environment
    """
    device = torch.device('cpu')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    # Only single environment for DDPG

    vec_env = DummyVecEnvWrapper(
        MujocoEnv('BipedalWalker-v2')
    ).instantiate(parallel_envs=1, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model_factory = DeterministicPolicyModelFactory(
        input_block=NormalizeObservationsFactory(input_shape=24),
        policy_backbone=MLPFactory(input_length=24, hidden_layers=[64, 64], normalization='layer'),
        value_backbone=MLPFactory(input_length=28, hidden_layers=[64, 64], normalization='layer')
    )

    # Reinforcer - an object managing the learning process
    reinforcer = BufferedOffPolicyIterationReinforcer(
        device=device,
        settings=BufferedOffPolicyIterationReinforcerSettings(
            rollout_steps=4,
            training_steps=1,
        ),
        environment=vec_env,
        algo=DeepDeterministicPolicyGradient(
            model_factory=model_factory,
            tau=0.01,
            discount_factor=0.99,
            max_grad_norm=0.5
        ),
        model=model_factory.instantiate(action_space=vec_env.action_space),
        env_roller=TransitionReplayEnvRoller(
            environment=vec_env,
            device=device,
            action_noise=OuNoise(std_dev=0.2, environment=vec_env),
            replay_buffer=CircularReplayBuffer(
                buffer_capacity=100,
                buffer_initial_size=100,
                num_envs=vec_env.num_envs,
                observation_space=vec_env.observation_space,
                action_space=vec_env.action_space
            ),
            normalize_returns=True,
            discount_factor=0.99
        ),
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
        MujocoEnv('BipedalWalker-v2', normalize_returns=True),
    ).instantiate(parallel_envs=8, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model_factory = StochasticPolicyModelSeparateFactory(
        input_block=NormalizeObservationsFactory(input_shape=24),
        policy_backbone=MLPFactory(input_length=24, hidden_layers=[32, 32]),
        value_backbone=MLPFactory(input_length=24, hidden_layers=[32])
    )

    # Reinforcer - an object managing the learning process
    reinforcer = OnPolicyIterationReinforcer(
        device=device,
        settings=OnPolicyIterationReinforcerSettings(
            number_of_steps=12,
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
            discount_factor=0.99,
            max_grad_norm=0.5,
            gae_lambda=1.0
        ),
        env_roller=StepEnvRoller(
            environment=vec_env,
            device=device,
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
    model_factory = QStochasticPolicyModelFactory(
        input_block=ImageToTensorFactory(),
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    )

    # Reinforcer - an object managing the learning process
    reinforcer = BufferedMixedPolicyIterationReinforcer(
        device=device,
        settings=BufferedMixedPolicyIterationReinforcerSettings(
            experience_replay=2,
            number_of_steps=12,
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
            discount_factor=0.99,
            max_grad_norm=10.0,
        ),
        env_roller=TrajectoryReplayEnvRoller(
            environment=vec_env,
            device=device,
            replay_buffer=CircularReplayBuffer(
                buffer_capacity=100,
                buffer_initial_size=100,
                num_envs=vec_env.num_envs,
                action_space=vec_env.action_space,
                observation_space=vec_env.observation_space,
                frame_stack_compensation=True,
                frame_history=4,
            )
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
