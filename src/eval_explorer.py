import os
os.environ["MUJOCO_GL"] = "egl"
import sys
from dataclasses import asdict
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange, pack
import wandb

from config import Config
from lexa import LEXA
from envs.env_factory import env_factory
from replay_buffer import ReplayBuffer
from utils import fix_seed, preprocess_obs, save_as_mp4, save_as_gif


base_path = Path(__file__).parents[1] / 'outputs'


def main(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    out_path = checkpoint_path.parent / 'eval' / checkpoint_path.name / 'explorer'
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(checkpoint_path)
    
    cfg: Config = ckpt['config']
    fix_seed(cfg.seed)
        
    env = env_factory(cfg.env.task, cfg.seed, cfg.env.img_size, cfg.env.action_repeat, cfg.env.time_limit)
    lexa = LEXA.load(ckpt)
    
    with torch.no_grad():
        for i in tqdm(range(10)):
            obs = env.reset()
            done = False
            video_data = [obs,]
            while not done:
                input_obs = preprocess_obs(obs)
                action = lexa.agent(input_obs, 'explorer', None, False)
                obs,  reward, done, info = env.step(action)
                video_data.append(obs)
            video_data = np.stack(video_data, axis=0)
            print(video_data.shape)
            save_as_mp4(video_data, out_path / f'{i}.mp4')
            save_as_gif(video_data, out_path / f'{i}.gif')


if __name__ == '__main__':
    args = sys.argv
    ckpt_path = args[1]
    main(ckpt_path)