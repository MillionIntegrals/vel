import numpy as np
import pandas as pd
import torch

from vel.api import ModelConfig, TrainingInfo
from vel.api.base import Storage, ModelFactory
from vel.rl.api.base import EnvFactory
from vel.openai.baselines.common.atari_wrappers import FrameStack


class EvaluateEnvCommand:
    """ Record environment playthrough as a game  """
    def __init__(self, model_config: ModelConfig, env_factory: EnvFactory, model_factory: ModelFactory,
                 storage: Storage, takes: int, frame_history: int,
                 sample_args: dict = None):
        self.model_config = model_config
        self.model_factory = model_factory
        self.env_factory = env_factory
        self.storage = storage
        self.takes = takes
        self.frame_history = frame_history
        self.sample_args = sample_args if sample_args is not None else {}

    def run(self):
        device = torch.device(self.model_config.device)

        env = FrameStack(self.env_factory.instantiate(preset='raw'), self.frame_history)
        model = self.model_factory.instantiate(action_space=env.action_space).to(device)

        training_info = TrainingInfo(start_epoch_idx=self.storage.last_epoch_idx(), run_name=self.model_config.run_name)
        self.storage.resume(training_info, model)

        model.eval()

        rewards = []
        lengths = []

        for i in range(self.takes):
            result = self.record_take(model, env, device, takenumber=i+1)
            rewards.append(result['r'])
            lengths.append(result['l'])

        print(pd.DataFrame({'lengths': lengths, 'rewards': rewards}).describe())

    @torch.no_grad()
    def record_take(self, model, env_instance, device, takenumber):
        frames = []

        observation = env_instance.reset()

        frames.append(env_instance.render('rgb_array'))

        print("Evaluating environment...")

        while True:
            observation_array = np.expand_dims(np.array(observation), axis=0)
            observation_tensor = torch.from_numpy(observation_array).to(device)
            actions = model.step(observation_tensor, **self.sample_args)['actions']

            observation, reward, done, epinfo = env_instance.step(actions.item())

            frames.append(env_instance.render('rgb_array'))

            if 'episode' in epinfo:
                # End of an episode
                return epinfo['episode']


def create(model_config, model, env, storage, takes, frame_history, sample_args=None):
    return EvaluateEnvCommand(
        model_config=model_config,
        model_factory=model,
        env_factory=env,
        storage=storage,
        frame_history=frame_history,
        takes=takes,
        sample_args=sample_args
    )
