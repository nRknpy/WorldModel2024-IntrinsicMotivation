from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from einops import rearrange

from .network import AchieverActor, AchieverCritic
from .worldmodel import WorldModel
from .achiever_reward import LatentDistanceReward
from .utils import compute_lambda_target


class Achiever(nn.Module):
    def __init__(self,
                 world_model: WorldModel,
                 instrinsic_reward,
                 action_dim,
                 z_dim,
                 num_classes,
                 h_dim,
                 emb_dim,
                 mlp_hidden_dim,
                 min_std,
                 discount,
                 lambda_,
                 actor_entropy_scale,
                 device):
        super(Achiever, self).__init__()
        
        self.world_model = world_model
        
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.discount = discount
        self.lambda_ = lambda_
        self.actor_entropy_scale = actor_entropy_scale
        self.device = device
        
        self.actor = AchieverActor(
            action_dim = action_dim,
            z_dim = z_dim,
            num_classes = num_classes,
            h_dim = h_dim,
            emb_dim = emb_dim,
            hidden_dim = mlp_hidden_dim,
            min_std = min_std
        )
        self.critic = AchieverCritic(
            z_dim = z_dim,
            num_classes = num_classes,
            h_dim = h_dim,
            emb_dim = emb_dim,
            hidden_dim = mlp_hidden_dim
        )
        self.target_critic = AchieverCritic(
            z_dim = z_dim,
            num_classes = num_classes,
            h_dim = h_dim,
            emb_dim = emb_dim,
            hidden_dim = mlp_hidden_dim
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.instrinsic_reward = instrinsic_reward
    
    def train(self, init_zs: torch.Tensor, init_hs: torch.Tensor, goal_observations: torch.Tensor, horison_length, num_positives, neg_sampling_factor, batch_seq_length):
        goal_observations = rearrange(goal_observations, 'b t c h w -> (t b) c h w')
        shuffle_idx = torch.randperm(goal_observations.shape[0])
        goal_observations = goal_observations[shuffle_idx]
        goal_embs = self.world_model.encoder(goal_observations)
        
        zs = init_zs.detach() # (batch_size * seq_length, z_dim * num_classes)
        hs = init_hs.detach() # (batch_size * seq_length, h_dim)
        
        imagined_zs = torch.empty(horison_length, *init_zs.shape, device=self.device)
        imagined_hs = torch.empty(horison_length, *init_hs.shape, device=self.device)
        imagined_action_log_probs = torch.empty(horison_length, init_zs.shape[0], device=self.device)
        imagined_action_entropys = torch.empty(horison_length, init_zs.shape[0], device=self.device)
        
        for t in range(horison_length):
            actions, action_log_probs, action_entropys = self.actor(zs.detach(), hs.detach(), goal_embs)
            
            with torch.no_grad():
                hs, zs = self.world_model.imagine(actions, zs, hs)
            
            imagined_hs[t] = hs.detach()
            imagined_zs[t] = zs.detach()
            imagined_action_log_probs[t] = action_log_probs
            imagined_action_entropys[t] = action_entropys
        
        flatten_hs = imagined_hs.view(-1, self.h_dim).detach() # (horison_length * batch_size * seq_length, h_dim)
        flatten_zs = imagined_zs.view(-1, self.z_dim * self.num_classes).detach() # (horison_length * batch_size * seq_length, z_dim * num_classes)
        
        with torch.no_grad():
            rewards = self.instrinsic_reward.compute_reward(flatten_zs, flatten_hs, goal_embs).view(horison_length, -1) # (horison_length, batch_size * seq_length)
            target_values = self.target_critic(flatten_zs, flatten_hs, goal_embs).view(horison_length, -1) # (horison_length, batch_size * seq_length)
        
        lambda_target = compute_lambda_target(rewards, self.discount, target_values, self.lambda_)
        
        objective = imagined_action_log_probs * ((lambda_target - target_values).detach())
        actor_loss = -torch.sum(torch.mean(objective + self.actor_entropy_scale * imagined_action_entropys, dim=1))
        
        value_mean = self.critic(flatten_zs.detach(), flatten_hs.detach()).view(horison_length, -1)
        value_dist = td.Independent(td.Normal(value_mean, 1),  1)
        critic_loss = -torch.mean(value_dist.log_prob(lambda_target.detach()).unsqueeze(-1))
        
        distance_estimator_loss = self.instrinsic_reward.train_distance_estimator(imagined_zs, imagined_hs, num_positives, neg_sampling_factor, batch_seq_length)
        
        return actor_loss, critic_loss, distance_estimator_loss, OrderedDict(ach_actor_loss=actor_loss.item(), ach_critic_loss=critic_loss.item(), distance_estimator_loss=distance_estimator_loss.item())
    
    def update_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
