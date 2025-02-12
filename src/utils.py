import random
import torch
import numpy as np
from PIL import Image
from einops import rearrange
import cv2
import imageio


def preprocess_obs(obs):
    height, width = obs.shape[0], obs.shape[1]
    obs = Image.fromarray(obs)
    obs = obs.convert("RGB")
    obs = np.array(obs).reshape(height, width, 3)
    obs = obs.astype(np.float32)
    obs = rearrange(obs, 'h w c -> c h w')
    normalized_obs = obs / 255.0 - 0.5
    return normalized_obs


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_as_mp4(video_data, filename="output.mp4", fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = video_data.shape[1:3]
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in video_data:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print(f"Saved mp4 to {filename}")

def save_as_gif(video_data, filename="output.gif", fps=30):
    duration = 1 / fps
    imageio.mimsave(filename, video_data, format="GIF", duration=duration)
    print(f"Saved GIF to {filename}")
