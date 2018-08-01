import cv2
import numpy as np
import os.path
import pathlib
import sys
import torch
import tqdm
import typing

from waterboy.api import ModelConfig
from waterboy.api.base import ModelAugmentor, Storage, Model
from waterboy.rl.api.base import VecEnvFactoryBase


class RecordMovieCommand:
    """ Record environment playthrough as a game  """
    def __init__(self, model_config: ModelConfig, vec_env: VecEnvFactoryBase, model: Model,
                 model_augmentors: typing.List[ModelAugmentor], storage: Storage,
                 videoname: str, takes: int, seed: int):
        self.model_config = model_config
        self.model = model
        self.model_augmentors = model_augmentors
        self.vec_env = vec_env
        self.storage = storage
        self.takes = takes
        self.videoname = videoname
        self.seed = seed

    def run(self):
        device = torch.device(self.model_config.device)

        env_instance = self.vec_env.instantiate_single(raw=True)

        model = self.model

        augmentor_dict = {'env': env_instance}

        for augmentor in self.model_augmentors:
            model = augmentor.augment(model, augmentor_dict)

        model = model.to(device)

        self.storage.resume_learning(model)

        model.eval()

        for i in range(self.takes):
            self.record_take(model, env_instance, device, takenumber=i+1)

    @torch.no_grad()
    def record_take(self, model, env_instance, device, takenumber):
        frames = []

        observation = env_instance.reset()

        frames.append(
            env_instance.unwrapped.render('rgb_array')
        )

        while True:
            observation_array = np.expand_dims(np.array(observation), axis=0)
            observation_tensor = torch.from_numpy(observation_array).to(device)
            actions, _, _ = model.step(observation_tensor)

            observation, reward, done, epinfo = env_instance.step(actions.item())

            frames.append(
                env_instance.unwrapped.render('rgb_array')
            )

            if 'episode' in epinfo:
                break

        takename = self.model_config.output_dir('videos', self.model_config.run_name, self.videoname.format(takenumber))
        pathlib.Path(os.path.dirname(takename)).mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video = cv2.VideoWriter(takename, fourcc, 30, (frames[0].shape[1], frames[0].shape[0]))

        for i in tqdm.trange(len(frames), file=sys.stdout):
            video.write(frames[i])

        video.release()
        print(f"Written {takename}")


def create(model_config, vec_env, model, model_augmentors, storage, takes, videoname, seed):
    return RecordMovieCommand(
        model_config=model_config,
        vec_env=vec_env,
        model=model,
        model_augmentors=model_augmentors,
        storage=storage,
        videoname=videoname,
        takes=takes,
        seed=seed
    )
