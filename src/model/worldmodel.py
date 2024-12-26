from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from einops import rearrange

from .network import RSSM, ConvEncoder, ConvDecoder, Discount


class WorldModel(nn.Module):
    def __init__(self,
                 img_size,
                 emb_dim,
                 action_dim,
                 z_dim,
                 num_classes,
                 h_dim,
                 hidden_dim,
                 num_layers_za2hidden,
                 num_layers_h2z,
                 mlp_hidden_dim,
                 min_std,
                 kl_balance_alpha,
                 kl_loss_scale,
                 device):
        super(WorldModel, self).__init__()
        
        self.img_size = img_size
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.num_layers_za2hidden = num_layers_za2hidden
        self.num_layers_h2z = num_layers_h2z
        self.mlp_hidden_dim = mlp_hidden_dim
        self.min_std = min_std
        self.kl_balance_alpha = kl_balance_alpha
        self.kl_loss_scale = kl_loss_scale
        self.device = device
        
        self.rssm = RSSM(
            z_dim = z_dim,
            num_classes = num_classes,
            h_dim = h_dim,
            hidden_dim = hidden_dim,
            emb_dim = emb_dim,
            action_dim = action_dim,
            num_layers_za2hidden = num_layers_za2hidden,
            num_layers_h2z = num_layers_h2z,
            min_std = min_std
        )
        self.encoder = ConvEncoder(
            input_size = img_size,
            emb_dim = emb_dim
        )
        self.decoder = ConvDecoder(
            img_size = img_size,
            z_dim = z_dim,
            num_classes = num_classes,
            h_dim = h_dim
        )
    
    def train(self, observations, actions):
        batch_size, seq_length, *_ = observations.shape
        observations = rearrange(observations, 'b t c h w -> t b c h w')
        actions = rearrange(actions, 'b t d -> t b d')
        
        embs = self.encoder(rearrange(observations, 't b c h w -> (t b) c h w'))
        embs = embs.view(seq_length, batch_size, -1)
        
        z = torch.zeros(batch_size, self.z_dim*self.num_classes, device=self.device)
        h = torch.zeros(batch_size, self.h_dim, device=self.device)
        
        zs = torch.empty(seq_length - 1, batch_size, self.z_dim*self.num_classes, device=self.device)
        hs = torch.empty(seq_length - 1, batch_size, self.h_dim, device=self.device)
        
        kl_loss = 0
        for t in range(seq_length - 1):
            h = self.rssm.recurrent(z, actions[t], h)
            next_prior, detach_next_prior = self.rssm.prior(h, detach=True)
            next_posterior, detach_next_posterior = self.rssm.posterior(h, embs[t+1], detach=True)
            z = next_posterior.rsample().flatten(1)
            hs[t] = h
            zs[t] = z
            kl_loss += self.kl_balance_alpha * torch.mean(kl_divergence(detach_next_posterior, next_prior)) + \
                       (1 - self.kl_balance_alpha) * torch.mean(kl_divergence(next_posterior, detach_next_prior))
        kl_loss = kl_loss / (seq_length - 1)
        
        flatten_hs = hs.view(-1, self.h_dim)
        flatten_zs = zs.view(-1, self.z_dim * self.num_classes)
        
        obs_dist = self.decoder(flatten_zs, flatten_hs)
        
        obs_loss = -torch.mean(obs_dist.log_prob(rearrange(observations[1:], 't b c h w -> (t b) c h w')))
        
        wm_loss = obs_loss + self.kl_loss_scale * kl_loss
        return wm_loss, (zs, hs), OrderedDict(wm_loss=wm_loss.item(), obs_loss=obs_loss.item(), kl_loss=kl_loss.item())

    def imagine(self, action, z, h):
        next_h = self.rssm.recurrent(z, action, h)
        next_prior = self.rssm.prior(next_h)
        next_z = next_prior.rsample().flatten(1)
        return next_h, next_z
