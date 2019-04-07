import attr
import numpy as np
import sys
import torch
import tqdm

from vel.api import Model, ModelFactory, TrainingInfo, EpochInfo, BatchInfo
from vel.rl.api import ReinforcerBase, ReinforcerFactory, VecEnvFactory, EnvRollerFactoryBase, EnvRollerBase, AlgoBase
from vel.rl.metrics import (
    FPSMetric, EpisodeLengthMetric, EpisodeRewardMetricQuantile,
    EpisodeRewardMetric, FramesMetric
)


@attr.s(auto_attribs=True)
class OnPolicyIterationReinforcerSettings:
    """ Settings dataclass for a policy gradient reinforcer """
    number_of_steps: int

    batch_size: int = 256
    experience_replay: int = 1
    stochastic_experience_replay: bool = False

    # For each experience replay loop, shuffle transitions to randomize gradient calculations
    # That means, disregarding actual trajectory order
    # Does not work with RNN policies
    shuffle_transitions: bool = True


class OnPolicyIterationReinforcer(ReinforcerBase):
    """
    A reinforcer that calculates on-policy environment rollouts and uses them to train policy directly.
    May split the sample into multiple batches and may replay batches a few times.
    """
    def __init__(self, device: torch.device, settings: OnPolicyIterationReinforcerSettings, model: Model,
                 algo: AlgoBase, env_roller: EnvRollerBase) -> None:
        self.device = device
        self.settings = settings

        self._trained_model = model.to(self.device)

        self.env_roller = env_roller
        self.algo = algo

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        my_metrics = [
            FramesMetric("frames"),
            FPSMetric("fps"),
            EpisodeRewardMetric('PMM:episode_rewards'),
            EpisodeRewardMetricQuantile('P09:episode_rewards', quantile=0.9),
            EpisodeRewardMetricQuantile('P01:episode_rewards', quantile=0.1),
            EpisodeLengthMetric("episode_length"),
        ]

        return my_metrics + self.algo.metrics() + self.env_roller.metrics()

    @property
    def model(self) -> Model:
        """ Model trained by this reinforcer """
        return self._trained_model

    def initialize_training(self, training_info: TrainingInfo, model_state=None, hidden_state=None):
        """ Prepare models for training """
        if model_state is not None:
            self.model.load_state_dict(model_state)
        else:
            self.model.reset_weights()

        self.algo.initialize(
            training_info=training_info, model=self.model, environment=self.env_roller.environment, device=self.device
        )

    def train_epoch(self, epoch_info: EpochInfo, interactive=True) -> None:
        """ Train model on an epoch of a fixed number of batch updates """
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

        1. Roll out the environmnent using current policy
        2. Use that rollout to train the policy
        """
        # Calculate environment rollout on the evaluation version of the model
        self.model.train()

        rollout = self.env_roller.rollout(batch_info, self.model, self.settings.number_of_steps)

        # Process rollout by the 'algo' (e.g. perform the advantage estimation)
        rollout = self.algo.process_rollout(batch_info, rollout)

        # Perform the training step
        # Algo will aggregate data into this list:
        batch_info['sub_batch_data'] = []

        if self.settings.shuffle_transitions:
            rollout = rollout.to_transitions()

        if self.settings.stochastic_experience_replay:
            # Always play experience at least once
            experience_replay_count = max(np.random.poisson(self.settings.experience_replay), 1)
        else:
            experience_replay_count = self.settings.experience_replay

        # Repeat the experience N times
        for i in range(experience_replay_count):
            # We may potentially need to split rollout into multiple batches
            if self.settings.batch_size >= rollout.frames():
                batch_result = self.algo.optimizer_step(
                    batch_info=batch_info,
                    device=self.device,
                    model=self.model,
                    rollout=rollout.to_device(self.device)
                )

                batch_info['sub_batch_data'].append(batch_result)
            else:
                # Rollout too big, need to split in batches
                for batch_rollout in rollout.shuffled_batches(self.settings.batch_size):
                    batch_result = self.algo.optimizer_step(
                        batch_info=batch_info,
                        device=self.device,
                        model=self.model,
                        rollout=batch_rollout.to_device(self.device)
                    )

                    batch_info['sub_batch_data'].append(batch_result)

        batch_info['frames'] = rollout.frames()
        batch_info['episode_infos'] = rollout.episode_information()

        # Even with all the experience replay, we count the single rollout as a single batch
        batch_info.aggregate_key('sub_batch_data')


class OnPolicyIterationReinforcerFactory(ReinforcerFactory):
    """ Vel factory class for the PolicyGradientReinforcer """
    def __init__(self, settings, parallel_envs: int, env_factory: VecEnvFactory, model_factory: ModelFactory,
                 algo: AlgoBase, env_roller_factory: EnvRollerFactoryBase, seed: int):
        self.settings = settings
        self.parallel_envs = parallel_envs

        self.env_factory = env_factory
        self.model_factory = model_factory
        self.algo = algo
        self.env_roller_factory = env_roller_factory
        self.seed = seed

    def instantiate(self, device: torch.device) -> ReinforcerBase:
        env = self.env_factory.instantiate(parallel_envs=self.parallel_envs, seed=self.seed)
        env_roller = self.env_roller_factory.instantiate(environment=env, device=device)
        model = self.model_factory.instantiate(action_space=env.action_space)

        return OnPolicyIterationReinforcer(device, self.settings, model, self.algo, env_roller)


def create(model_config, model, vec_env, algo, env_roller, parallel_envs, number_of_steps,
           batch_size=256, experience_replay=1, stochastic_experience_replay=False, shuffle_transitions=True):
    """ Vel factory function """
    settings = OnPolicyIterationReinforcerSettings(
        number_of_steps=number_of_steps,
        batch_size=batch_size,
        experience_replay=experience_replay,
        stochastic_experience_replay=stochastic_experience_replay,
        shuffle_transitions=shuffle_transitions
    )

    return OnPolicyIterationReinforcerFactory(
        settings=settings,
        parallel_envs=parallel_envs,
        env_factory=vec_env,
        model_factory=model,
        algo=algo,
        env_roller_factory=env_roller,
        seed=model_config.seed
    )
