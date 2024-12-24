import torch
import torch.nn as nn

from .model.worldmodel import WorldModel
from .model.explorer import Explorer
from .model.achiever import Achiever


class LEXA(nn.Module):
    def __init__(self, cfg):
        super(LEXA, self).__init__()
        
        