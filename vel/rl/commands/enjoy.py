import numpy as np
import torch
import typing
import time

from vel.api import ModelConfig, TrainingInfo, Storage, ModelFactory
from vel.rl.api import VecEnvFactory


class EnjoyCommand:
    """ Play render("human") in a loop for a human to enjoy """

    def __init__(self, model_config: ModelConfig, model_factory: ModelFactory, vec_env_factory: VecEnvFactory,
                 storage: Storage, fps: float, sample_args: typing.Optional[dict]):
        self.model_config = model_config
        self.model_factory = model_factory
        self.vec_env_factory = vec_env_factory
        self.storage = storage

        self.fps = fps

        self.sample_args = sample_args if sample_args is not None else {}

    def run(self):
        """ Run the command """
        device = self.model_config.torch_device()

        env = self.vec_env_factory.instantiate_single(preset='record', seed=self.model_config.seed)
        model = self.model_factory.instantiate(action_space=env.action_space).to(device)

        training_info = TrainingInfo(
            start_epoch_idx=self.storage.last_epoch_idx(),
            run_name=self.model_config.run_name
        )

        self.storage.load(training_info, model)

        model.eval()

        self.run_model(model, env, device)

    @torch.no_grad()
    def run_model(self, model, environment, device):
        observation = environment.reset()
        current_time = time.time()

        seconds_per_frame = 1.0 / self.fps

        if model.is_recurrent:
            hidden_state = model.zero_state(1).to(device)

        while True:
            observation_array = np.expand_dims(np.array(observation), axis=0)
            observation_tensor = torch.from_numpy(observation_array).to(device)

            if model.is_recurrent:
                output = model.step(observation_tensor, hidden_state, **self.sample_args)
                hidden_state = output['state']
                actions = output['actions']
            else:
                actions = model.step(observation_tensor, **self.sample_args)['actions']

            actions = actions.detach().cpu().numpy()

            observation, reward, done, epinfo = environment.step(actions[0])

            environment.render('human')

            frame_time = time.time()

            if (frame_time - current_time) < seconds_per_frame:
                time.sleep(seconds_per_frame - (frame_time - current_time))

            current_time = frame_time

            if 'episode' in epinfo:
                # End of an episode
                break


def create(model_config, model, vec_env, storage, fps=30.0, sample_args=None):
    """ Vel factory function """
    return EnjoyCommand(
        model_config, model, vec_env, storage, float(fps), sample_args
    )
