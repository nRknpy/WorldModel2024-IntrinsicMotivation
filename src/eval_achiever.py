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
    out_path = checkpoint_path.parent / 'eval' / checkpoint_path.name / 'achiever'
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(checkpoint_path)
    
    cfg: Config = ckpt['config']
    fix_seed(cfg.seed)
        
    env = env_factory(cfg.env.task, cfg.seed, cfg.env.img_size, cfg.env.action_repeat, cfg.env.time_limit)
    lexa = LEXA.load(ckpt)
    
    with torch.no_grad():
        for goal_idx in tqdm(env.goals):
            env.set_goal_idx(goal_idx)
            obs = env.reset()
            goal = env.get_goal_obs()
            input_goal = preprocess_obs(goal)
            done = False
            frame, _ = pack([obs, goal], 'h * c')
            video_data = [frame,]
            while not done:
                input_obs = preprocess_obs(obs)
                action = lexa.agent(input_obs, 'achiever', input_goal, False)
                obs,  reward, done, info = env.step(action)
                frame, _ = pack([obs, goal], 'h * c')
                video_data.append(frame)
            video_data = np.stack(video_data, axis=0)
            print(video_data.shape)
            save_as_mp4(video_data, out_path / f'{goal_idx}.mp4')
            save_as_gif(video_data, out_path / f'{goal_idx}.gif')


if __name__ == '__main__':
    args = sys.argv
    ckpt_path = args[1]
    main(ckpt_path)