import attr
import sys
import tqdm

import torch

from vel.api import TrainingInfo, EpochInfo, BatchInfo, Model, ModelFactory
from vel.openai.baselines.common.vec_env import VecEnv
from vel.rl.api import (
    ReinforcerBase, ReinforcerFactory, ReplayEnvRollerBase, AlgoBase, VecEnvFactory, ReplayEnvRollerFactoryBase
)
from vel.rl.metrics import (
    FPSMetric, EpisodeLengthMetric, EpisodeRewardMetricQuantile, EpisodeRewardMetric, FramesMetric,
)


@attr.s(auto_attribs=True)
class BufferedOffPolicyIterationReinforcerSettings:
    """ Settings class for deep Q-Learning """
    # How many steps to roll out for each env
    rollout_steps: int

    # How many steps to generate as a roll out from replay buffer
    training_steps: int
    # How many times to roll out
    training_rounds: int = 1


class BufferedOffPolicyIterationReinforcer(ReinforcerBase):
    """
    An off-policy reinforcer that rolls out environment and stores transitions in a buffer.
    Afterwards, it samples experience batches from this buffer to train the policy.
    """
    def __init__(self, device: torch.device, settings: BufferedOffPolicyIterationReinforcerSettings,
                 environment: VecEnv, model: Model, algo: AlgoBase, env_roller: ReplayEnvRollerBase):
        self.device = device
        self.settings = settings
        self.environment = environment

        self._trained_model = model.to(self.device)
        self.algo = algo

        self.env_roller = env_roller

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        my_metrics = [
            FramesMetric("frames"),
            FPSMetric("fps"),
            EpisodeRewardMetric('PMM:episode_rewards'),
            EpisodeRewardMetricQuantile('P09:episode_rewards', quantile=0.9),
            EpisodeRewardMetricQuantile('P01:episode_rewards', quantile=0.1),
            EpisodeLengthMetric("episode_length")
        ]

        return my_metrics + self.algo.metrics() + self.env_roller.metrics()

    @property
    def model(self) -> Model:
        return self._trained_model

    def initialize_training(self, training_info: TrainingInfo, model_state=None, hidden_state=None):
        """ Prepare models for training """
        if model_state is not None:
            self.model.load_state_dict(model_state)
        else:
            self.model.reset_weights()

        self.algo.initialize(
            training_info=training_info, model=self.model, environment=self.environment, device=self.device
        )

    def train_epoch(self, epoch_info: EpochInfo, interactive=True) -> None:
        """ Train model for a single epoch  """
        epoch_info.on_epoch_begin()

        if interactive:
            iterator = tqdm.trange(epoch_info.batches_per_epoch, file=sys.stdout, desc="Training", unit="batch")
        else:
            iterator = range(epoch_info.batches_per_epoch)

        for batch_idx in iterator:
            batch_info = BatchInfo(epoch_info, batch_idx)

            batch_info.on_batch_begin()
            self.train_batch(batch_info)
            batch_info.on_batch_end()

        epoch_info.result_accumulator.freeze_results()
        epoch_info.on_epoch_end()

    def train_batch(self, batch_info: BatchInfo) -> None:
        """
        Batch - the most atomic unit of learning.

        For this reinforforcer, that involves:

        1. Roll out environment and store out experience in the buffer
        2. Sample the buffer and train the algo on sample batch
        """
        # For each reinforcer batch:

        # 1. Roll out environment and store out experience in the buffer
        self.roll_out_and_store(batch_info)

        # 2. Sample the buffer and train the algo on sample batch
        self.train_on_replay_memory(batch_info)

    def roll_out_and_store(self, batch_info):
        """ Roll out environment and store result in the replay buffer """
        self.model.train()

        if self.env_roller.is_ready_for_sampling():
            rollout = self.env_roller.rollout(batch_info, self.model, self.settings.rollout_steps).to_device(self.device)

            # Store some information about the rollout, no training phase
            batch_info['frames'] = rollout.frames()
            batch_info['episode_infos'] = rollout.episode_information()
        else:
            frames = 0
            episode_infos = []

            with tqdm.tqdm(desc="Populating memory", total=self.env_roller.initial_memory_size_hint()) as pbar:
                while not self.env_roller.is_ready_for_sampling():
                    rollout = self.env_roller.rollout(batch_info, self.model, self.settings.rollout_steps).to_device(self.device)

                    new_frames = rollout.frames()
                    frames += new_frames
                    episode_infos.extend(rollout.episode_information())

                    pbar.update(new_frames)

            # Store some information about the rollout, no training phase
            batch_info['frames'] = frames
            batch_info['episode_infos'] = episode_infos

    def train_on_replay_memory(self, batch_info):
        """ Train agent on a memory gotten from replay buffer """
        self.model.train()

        # Algo will aggregate data into this list:
        batch_info['sub_batch_data'] = []

        for i in range(self.settings.training_rounds):
            sampled_rollout = self.env_roller.sample(batch_info, self.model, self.settings.training_steps)

            batch_result = self.algo.optimizer_step(
                batch_info=batch_info,
                device=self.device,
                model=self.model,
                rollout=sampled_rollout.to_device(self.device)
            )

            self.env_roller.update(rollout=sampled_rollout, batch_info=batch_result)

            batch_info['sub_batch_data'].append(batch_result)

        batch_info.aggregate_key('sub_batch_data')


class BufferedOffPolicyIterationReinforcerFactory(ReinforcerFactory):
    """ Factory class for the DQN reinforcer """

    def __init__(self, settings, env_factory: VecEnvFactory, model_factory: ModelFactory,
                 algo: AlgoBase, env_roller_factory: ReplayEnvRollerFactoryBase, parallel_envs: int, seed: int):
        self.settings = settings

        self.env_factory = env_factory
        self.model_factory = model_factory
        self.algo = algo
        self.env_roller_factory = env_roller_factory
        self.parallel_envs = parallel_envs
        self.seed = seed

    def instantiate(self, device: torch.device) -> BufferedOffPolicyIterationReinforcer:
        env = self.env_factory.instantiate(parallel_envs=self.parallel_envs, seed=self.seed)
        env_roller = self.env_roller_factory.instantiate(env, device)
        model = self.model_factory.instantiate(action_space=env.action_space)

        return BufferedOffPolicyIterationReinforcer(
            device=device,
            settings=self.settings,
            environment=env,
            model=model,
            algo=self.algo,
            env_roller=env_roller
        )


def create(model_config, vec_env, model, algo, env_roller, parallel_envs: int,
           rollout_steps: int, training_steps: int, training_rounds: int=1):
    """ Vel factory function """
    settings = BufferedOffPolicyIterationReinforcerSettings(
        rollout_steps=rollout_steps,
        training_steps=training_steps,
        training_rounds=training_rounds
    )

    return BufferedOffPolicyIterationReinforcerFactory(
        settings=settings,
        env_factory=vec_env,
        model_factory=model,
        algo=algo,
        env_roller_factory=env_roller,
        parallel_envs=parallel_envs,
        seed=model_config.seed
    )
