import attr
import numpy as np
import sys
import torch
import tqdm

from vel.api import ModelFactory, TrainingInfo, EpochInfo, BatchInfo
from vel.rl.api import (
    Reinforcer, ReinforcerFactory, VecEnvFactory, EnvRollerFactoryBase, EnvRollerBase,
    RlPolicy
)
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


class OnPolicyIterationReinforcer(Reinforcer):
    """
    A reinforcer that calculates on-policy environment rollouts and uses them to train policy directly.
    May split the sample into multiple batches and may replay batches a few times.
    """
    def __init__(self, device: torch.device, settings: OnPolicyIterationReinforcerSettings, policy: RlPolicy,
                 env_roller: EnvRollerBase) -> None:
        self.device = device
        self.settings = settings
        self.env_roller = env_roller

        self._model: RlPolicy = policy.to(self.device)

    @property
    def policy(self) -> RlPolicy:
        """ Model trained by this reinforcer """
        return self._model

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

        return my_metrics + self.env_roller.metrics() + self.policy.metrics()

    def initialize_training(self, training_info: TrainingInfo, model_state=None, hidden_state=None):
        """ Prepare models for training """
        if model_state is not None:
            self.policy.load_state_dict(model_state)
        else:
            self.policy.reset_weights()

    def train_epoch(self, epoch_info: EpochInfo, interactive=True) -> None:
        """ Train model on an epoch of a fixed number of batch updates """
        epoch_info.on_epoch_begin()

        if interactive:
            iterator = tqdm.trange(epoch_info.batches_per_epoch, file=sys.stdout, desc="Training", unit="batch")
        else:
            iterator = range(epoch_info.batches_per_epoch)

        for batch_idx in iterator:
            batch_info = BatchInfo(epoch_info, batch_idx)

            batch_info.on_batch_begin('train')
            self.train_batch(batch_info)
            batch_info.on_batch_end('train')

        epoch_info.result_accumulator.freeze_results()
        epoch_info.on_epoch_end()

    def train_batch(self, batch_info: BatchInfo) -> None:
        """
        Batch - the most atomic unit of learning.

        For this reinforcer, that involves:

        1. Roll out the environmnent using current policy
        2. Use that rollout to train the policy
        """
        rollout = self.env_roller.rollout(batch_info, self.settings.number_of_steps)

        # Preprocessing of the rollout for this algorithm
        rollout = self.policy.process_rollout(rollout)

        # Perform the training step
        # Algo will aggregate data into this list:
        batch_info['sub_batch_data'] = []

        if self.settings.shuffle_transitions:
            rollout = rollout.to_transitions()

        if self.settings.stochastic_experience_replay:
            # Always play experience at least once
            experience_replay_count = 1 + np.random.poisson(self.settings.experience_replay - 1)
        else:
            experience_replay_count = self.settings.experience_replay

        self.policy.train()

        # Repeat the experience N times
        for i in range(experience_replay_count):
            # We may potentially need to split rollout into multiple batches
            if self.settings.batch_size >= rollout.frames():
                metrics = self.policy.optimize(
                    batch_info=batch_info,
                    rollout=rollout.to_device(self.device),
                )

                batch_info['sub_batch_data'].append(metrics)
            else:
                # Rollout too big, need to split in batches
                for batch_rollout in rollout.shuffled_batches(self.settings.batch_size):

                    metrics = self.policy.optimize(
                        batch_info=batch_info,
                        rollout=batch_rollout.to_device(self.device),
                    )

                    batch_info['sub_batch_data'].append(metrics)

        batch_info['frames'] = rollout.frames()
        batch_info['episode_infos'] = rollout.episode_information()

        # Even with all the experience replay, we count the single rollout as a single batch
        batch_info.aggregate_key('sub_batch_data')


class OnPolicyIterationReinforcerFactory(ReinforcerFactory):
    """ Vel factory class for the PolicyGradientReinforcer """
    def __init__(self, settings, parallel_envs: int, env_factory: VecEnvFactory, model_factory: ModelFactory,
                 env_roller_factory: EnvRollerFactoryBase, seed: int):
        self.settings = settings
        self.parallel_envs = parallel_envs

        self.env_factory = env_factory
        self.model_factory = model_factory
        self.env_roller_factory = env_roller_factory
        self.seed = seed

    def instantiate(self, device: torch.device) -> Reinforcer:
        env = self.env_factory.instantiate(parallel_envs=self.parallel_envs, seed=self.seed)
        policy = self.model_factory.instantiate(action_space=env.action_space, observation_space=env.observation_space)
        env_roller = self.env_roller_factory.instantiate(environment=env, policy=policy, device=device)
        return OnPolicyIterationReinforcer(device, self.settings, policy, env_roller)


def create(model_config, model, vec_env, env_roller, parallel_envs, number_of_steps,
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
        env_roller_factory=env_roller,
        seed=model_config.seed
    )
