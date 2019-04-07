import numpy as np
import pandas as pd
import torch
import tqdm
import typing

from vel.api import ModelConfig, TrainingInfo, Storage, ModelFactory
from vel.rl.api import VecEnvFactory


class EvaluateEnvCommand:
    """ Record environment playthrough as a game  """
    def __init__(self, model_config: ModelConfig, env_factory: VecEnvFactory, model_factory: ModelFactory,
                 storage: Storage, parallel_envs: int, action_noise: typing.Optional[ModelFactory],  takes: int, sample_args: dict = None):
        self.model_config = model_config
        self.model_factory = model_factory
        self.env_factory = env_factory
        self.storage = storage
        self.takes = takes
        self.parallel_envs = parallel_envs
        self.action_noise_factory = action_noise

        self.sample_args = sample_args if sample_args is not None else {}

    @torch.no_grad()
    def run(self):
        device = self.model_config.torch_device()

        env = self.env_factory.instantiate(parallel_envs=self.parallel_envs, preset='record', seed=self.model_config.seed)
        model = self.model_factory.instantiate(action_space=env.action_space).to(device)

        if self.action_noise_factory is not None:
            action_noise = self.action_noise_factory.instantiate(environment=env).to(device)
        else:
            action_noise = None

        training_info = TrainingInfo(
            start_epoch_idx=self.storage.last_epoch_idx(), run_name=self.model_config.run_name
        )

        model_state, hidden_state = self.storage.load(training_info)
        model.load_state_dict(model_state)

        print("Loading model trained for {} epochs".format(training_info.start_epoch_idx))

        model.eval()

        episode_rewards = []
        episode_lengths = []

        observations = env.reset()
        observations_tensor = torch.from_numpy(observations).to(device)

        if model.is_recurrent:
            hidden_state = model.zero_state(observations.shape[0]).to(device)

        with tqdm.tqdm(total=self.takes) as progress_bar:
            while len(episode_rewards) < self.takes:
                if model.is_recurrent:
                    output = model.step(observations_tensor, hidden_state, **self.sample_args)
                    hidden_state = output['state']
                    actions = output['actions']
                else:
                    actions = model.step(observations_tensor, **self.sample_args)['actions']

                if action_noise is not None:
                    actions = action_noise(actions)

                observations, rewards, dones, infos = env.step(actions.cpu().numpy())
                observations_tensor = torch.from_numpy(observations).to(device)

                for info in infos:
                    if 'episode' in info:
                        episode_rewards.append(info['episode']['r'])
                        episode_lengths.append(info['episode']['l'])
                        progress_bar.update(1)

                if model.is_recurrent:
                    # Zero state belongiong to finished episodes
                    dones_tensor = torch.from_numpy(dones.astype(np.float32)).to(device)
                    hidden_state = hidden_state * (1.0 - dones_tensor.unsqueeze(-1))

        print(pd.DataFrame({'lengths': episode_lengths, 'rewards': episode_rewards}).describe())


def create(model_config, model, vec_env, storage, takes, parallel_envs, action_noise=None, sample_args=None):
    """ Vel factory function """
    return EvaluateEnvCommand(
        model_config=model_config,
        model_factory=model,
        env_factory=vec_env,
        parallel_envs=parallel_envs,
        action_noise=action_noise,
        storage=storage,
        takes=takes,
        sample_args=sample_args
    )
