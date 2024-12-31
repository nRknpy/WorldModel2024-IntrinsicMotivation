from typing import Literal
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .network import ExprolerStatePredictor


class EmsembleReward(nn.Module):
    def __init__(self,
                 z_dim,
                 num_classes,
                 h_dim,
                 min_std,
                 mlp_hidden_dim,
                 device,
                 num_emsembles = 10,
                 offset = 1,
                 target_mode: Literal['z', 'h', 'zh'] = 'z'):
        super(EmsembleReward, self).__init__()
        
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.min_std = min_std
        self.num_emsembles = num_emsembles
        self.offset = offset
        self.target_mode = target_mode
        self.mlp_hidden_dim = mlp_hidden_dim
        self.device = device
        
        if target_mode == 'z':
            self.target_dim = z_dim * num_classes
        elif target_mode == 'h':
            self.target_dim = h_dim
        elif target_mode == 'zh':
            self.target_dim = z_dim * num_classes + h_dim
        
        self.emsembles = nn.ModuleList()
        for _ in range(num_emsembles):
            self.emsembles.append(
                ExprolerStatePredictor(
                    z_dim = z_dim,
                    num_classes = num_classes,
                    h_dim = h_dim,
                    min_std = min_std,
                    hidden_dim = mlp_hidden_dim,
                    target_dim = self.target_dim
                )
            )
    
    def compute_reward(self, z, h):
        preds = torch.empty(self.num_emsembles, z.shape[0], self.target_dim, device=self.device)
        for n in range(self.num_emsembles):
            f = self.emsembles[n]
            preds[n] = f(z, h).mean
        var = torch.std(preds, dim=0)
        reward = torch.mean(var, dim=1)
        return reward
    
    def train(self, zs, hs):
        if self.target_mode == 'z':
            target = zs
        elif self.target_mode == 'h':
            target = hs
        elif self.target_mode == 'zh':
            target = torch.concat([zs, hs], dim=2)
        
        input_zs = rearrange(zs[:-self.offset].detach(), 't b d -> (t b) d')
        input_hs = rearrange(hs[:-self.offset].detach(), 't b d -> (t b) d')
        target = rearrange(target[self.offset:].detach(), 't b d -> (t b) d')
        
        loss = 0
        for f in self.emsembles:
            dist = f(input_zs, input_hs)
            loss += -torch.mean(dist.log_prob(target))
        return loss, OrderedDict(emsemble_loss=loss.item())
