import torch
import pandas as pd
import numpy as np

from vel.rl.env.classic_atari import ClassicAtariEnv

from vel.rl.models.policy_gradient_model import PolicyGradientModelFactory
from vel.rl.models.backbone.nature_cnn import NatureCnnFactory

from vel.openai.baselines.common.atari_wrappers import FrameStack


def breakout_a2c_evaluate(checkpoint_file_path, takes=10):
    model_checkpoint = torch.load(checkpoint_file_path)
    device = torch.device('cuda:0')

    env = FrameStack(
        ClassicAtariEnv('BreakoutNoFrameskip-v4').instantiate(preset='raw'), k=4
    )

    model = PolicyGradientModelFactory(
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    ).instantiate(action_space=env.action_space)

    model.load_state_dict(model_checkpoint)
    model = model.to(device)

    model.eval()

    rewards = []
    lengths = []

    for i in range(takes):
        result = record_take(model, env, device)
        rewards.append(result['r'])
        lengths.append(result['l'])

    print(pd.DataFrame({'lengths': lengths, 'rewards': rewards}).describe())


@torch.no_grad()
def record_take(model, env_instance, device):
    frames = []

    observation = env_instance.reset()

    frames.append(env_instance.render('rgb_array'))

    print("Evaluating environment...")

    while True:
        observation_array = np.expand_dims(np.array(observation), axis=0)
        observation_tensor = torch.from_numpy(observation_array).to(device)
        actions = model.step(observation_tensor, argmax_sampling=True)['actions']

        observation, reward, done, epinfo = env_instance.step(actions.item())

        frames.append(env_instance.render('rgb_array'))

        if 'episode' in epinfo:
            # End of an episode
            return epinfo['episode']


if __name__ == '__main__':
    breakout_a2c_evaluate("checkpoint_00001375.data", takes=2)
