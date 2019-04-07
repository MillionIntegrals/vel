import cv2
import numpy as np
import os.path
import pathlib
import sys
import torch
import tqdm
import typing

from vel.api import ModelConfig, TrainingInfo, Storage, ModelFactory
from vel.rl.api import VecEnvFactory


class RecordMovieCommand:
    """ Record environment playthrough as a game  """
    def __init__(self, model_config: ModelConfig, env_factory: VecEnvFactory, model_factory: ModelFactory,
                 storage: Storage, videoname: str, takes: int, fps: int, sample_args: typing.Optional[dict] = None):
        self.model_config = model_config
        self.model_factory = model_factory
        self.env_factory = env_factory
        self.storage = storage
        self.takes = takes
        self.videoname = videoname
        self.sample_args = sample_args if sample_args is not None else {}
        self.fps = fps

    def run(self):
        device = self.model_config.torch_device()

        env = self.env_factory.instantiate_single(preset='record', seed=self.model_config.seed)
        model = self.model_factory.instantiate(action_space=env.action_space).to(device)

        training_info = TrainingInfo(
            start_epoch_idx=self.storage.last_epoch_idx(),
            run_name=self.model_config.run_name
        )

        model_state, hidden_state = self.storage.load(training_info)
        model.load_state_dict(model_state)

        model.eval()

        for i in range(self.takes):
            self.record_take(model, env, device, take_number=i + 1)

    @torch.no_grad()
    def record_take(self, model, env_instance, device, take_number):
        """ Record a single movie and store it on hard drive """
        frames = []

        observation = env_instance.reset()

        if model.is_recurrent:
            hidden_state = model.zero_state(1).to(device)

        frames.append(env_instance.render('rgb_array'))

        print("Evaluating environment...")

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

            observation, reward, done, epinfo = env_instance.step(actions[0])

            frames.append(env_instance.render('rgb_array'))

            if 'episode' in epinfo:
                # End of an episode
                break

        takename = self.model_config.output_dir('videos', self.model_config.run_name, self.videoname.format(take_number))
        pathlib.Path(os.path.dirname(takename)).mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video = cv2.VideoWriter(takename, fourcc, self.fps, (frames[0].shape[1], frames[0].shape[0]))

        for i in tqdm.trange(len(frames), file=sys.stdout):
            video.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))

        video.release()
        print("Written {}".format(takename))


def create(model_config, model, vec_env, storage, takes, videoname, fps=30, sample_args=None):
    """ Vel factory function """
    return RecordMovieCommand(
        model_config=model_config,
        model_factory=model,
        env_factory=vec_env,
        storage=storage,
        videoname=videoname,
        takes=takes,
        fps=fps,
        sample_args=sample_args
    )
