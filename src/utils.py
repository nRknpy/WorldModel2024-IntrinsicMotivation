import random
import torch
import numpy as np
from PIL import Image
from einops import rearrange


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
