import cv2
import numpy as np
import os.path
import pathlib
import sys
import torch
import tqdm

from vel.api import ModelConfig
from vel.api.base import Storage, ModelFactory
from vel.rl.api.base import VecEnvFactory


class RecordMovieCommand:
    """ Record environment playthrough as a game  """
    def __init__(self, model_config: ModelConfig, vec_env: VecEnvFactory, model_factory: ModelFactory,
                 storage: Storage, videoname: str, takes: int, maxframes: int, seed: int):
        self.model_config = model_config
        self.model_factory = model_factory
        self.vec_env = vec_env
        self.storage = storage
        self.takes = takes
        self.videoname = videoname
        self.maxframes = maxframes
        self.seed = seed

    def run(self):
        device = torch.device(self.model_config.device)

        env_instance = self.vec_env.instantiate_single(preset='raw')
        model = self.model_factory.instantiate(action_space=env_instance.action_space).to(device)

        self.storage.resume_learning(model)

        model.eval()

        for i in range(self.takes):
            self.record_take(model, env_instance, device, takenumber=i+1)

    @torch.no_grad()
    def record_take(self, model, env_instance, device, takenumber):
        frames = []

        observation = env_instance.reset()

        frames.append(env_instance.render('rgb_array'))

        for i in range(self.maxframes):
            observation_array = np.expand_dims(np.array(observation), axis=0)
            observation_tensor = torch.from_numpy(observation_array).to(device)
            actions, _, _ = model.step(observation_tensor)

            observation, reward, done, epinfo = env_instance.step(actions.item())

            frames.append(env_instance.render('rgb_array'))

            if 'episode' in epinfo:
                # End of an episode
                break

        takename = self.model_config.output_dir('videos', self.model_config.run_name, self.videoname.format(takenumber))
        pathlib.Path(os.path.dirname(takename)).mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video = cv2.VideoWriter(takename, fourcc, 30, (frames[0].shape[1], frames[0].shape[0]))

        for i in tqdm.trange(len(frames), file=sys.stdout):
            video.write(frames[i])

        video.release()
        print(f"Written {takename}")


def create(model_config, vec_env, model, storage, takes, videoname, seed, maxframes=10000):
    return RecordMovieCommand(
        model_config=model_config,
        vec_env=vec_env,
        model_factory=model,
        storage=storage,
        videoname=videoname,
        takes=takes,
        maxframes=maxframes,
        seed=seed
    )
