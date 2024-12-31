from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .network import AchieverDistanceEstimator, State2Emb


class LatentDistanceReward(nn.Module):
    def __init__(self,
                 state2emb: State2Emb,
                 z_dim,
                 num_classes,
                 h_dim,
                 emb_dim,
                 mlp_hidden_dim,
                 device):
        super(LatentDistanceReward, self).__init__()
        
        self.state2emb = state2emb
        
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.device = device
        
        self.distance_estimator = AchieverDistanceEstimator(
            emb_dim = emb_dim,
            hidden_dim = mlp_hidden_dim
        )
    
    def imagine_compute_reward(self, current_z, current_h, goal_z, goal_h):
        current_emb = self.state2emb(current_z, current_h).mean
        goal_emb = self.state2emb(goal_z, goal_h).mean
        distance = self.distance_estimator(current_emb, goal_emb)
        return -distance
    
    def compute_reward(self, z, h, goal_emb):
        current_emb = self.state2emb(z, h).mean
        distance = self.distance_estimator(current_emb, goal_emb)
        return -distance
    
    def train(self, zs, hs, num_positives, neg_sampling_factor, horison_length, batch_size, batch_length):
        def get_future_goal_idxs(seq_len, bs):
            cur_idx_list = []
            goal_idx_list = []
            for cur_idx in range(seq_len):
                for goal_idx in range(cur_idx, seq_len):
                    cur_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*cur_idx, np.arange(bs).reshape(-1,1)], axis = -1))
                    goal_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*goal_idx, np.arange(bs).reshape(-1,1)], axis = -1))
            
            return np.concatenate(cur_idx_list,0), np.concatenate(goal_idx_list,0)

        def get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs):
            cur_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
            goal_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
            for i in range(num_negs):
                goal_idxs[i,1] = np.random.choice([j for j in range(bs) if j//batch_length != cur_idxs[i,1]//batch_length])
            return cur_idxs, goal_idxs
        
        zs, hs = zs.detach(), hs.detach()
        
        current_idxs, goal_idxs = get_future_goal_idxs(horison_length, batch_size)
        idx = np.random.choice(np.arange(len(current_idxs)), num_positives, replace=False)
        current_idx, goal_idx = current_idxs[idx], goal_idxs[idx]
        current_zs, current_hs = zs[current_idx[:,0], current_idx[:,1]], hs[current_idx[:,0], current_idx[:,1]]
        goal_zs, goal_hs = zs[goal_idx[:,0], goal_idx[:,1]], hs[goal_idx[:,0], goal_idx[:,1]]
        current_embs, goal_embs = self.state2emb(current_zs, current_hs).mean, self.state2emb(goal_zs, goal_hs).mean
        target_distance = torch.from_numpy((goal_idx[:,0] - current_idx[:,0]) / horison_length).to(self.device, dtype=zs.dtype).unsqueeze(1)
        pred_distance = self.distance_estimator(current_embs.detach(), goal_embs.detach())
        loss = F.mse_loss(pred_distance, target_distance)
        
        num_negatives = int(num_positives * neg_sampling_factor)
        current_idx, goal_idx = get_future_goal_idxs_neg_sampling(num_negatives, horison_length, batch_size)
        current_zs, current_hs = zs[current_idx[:,0], current_idx[:,1]], hs[current_idx[:,0], current_idx[:,1]]
        goal_zs, goal_hs = zs[goal_idx[:,0], goal_idx[:,1]], hs[goal_idx[:,0], goal_idx[:,1]]
        current_embs, goal_embs = self.state2emb(current_zs, current_hs).mean, self.state2emb(goal_zs, goal_hs).mean
        target_distance = torch.ones(num_negatives, 1, device=self.device) * horison_length
        pred_distance = self.distance_estimator(current_embs.detach(), goal_embs.detach())
        loss += F.mse_loss(pred_distance, target_distance)
        return loss
