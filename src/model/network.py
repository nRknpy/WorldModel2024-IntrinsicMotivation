from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, OneHotCategoricalStraightThrough, Independent, Bernoulli

from .utils import TruncatedNormal


class RSSM(nn.Module):
    def __init__(self,
                 z_dim=30,
                 num_classes=20,
                 h_dim=200,
                 hidden_dim=200,
                 emb_dim=32,
                 action_dim=9,
                 num_layers_za2hidden=1,
                 num_layers_h2z=1,
                 min_std=0.1):
        super(RSSM, self).__init__()
        
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.action_dim = action_dim
        self.num_layers_za2hidden = num_layers_za2hidden
        self.num_layers_h2z = num_layers_h2z
        self.num_classes = num_classes
        self.min_std = min_std
        
        self.za2hidden = nn.Sequential(
            *([nn.Sequential(nn.Linear(self.z_dim * self.num_classes + self.action_dim, self.hidden_dim), nn.ELU())] + \
            [nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ELU()) for _ in range(self.num_layers_za2hidden - 1)])
        )
        self.transition = nn.GRUCell(self.hidden_dim, self.h_dim)
        
        self.prior_hidden = nn.Sequential(
            *([nn.Sequential(nn.Linear(self.h_dim, self.hidden_dim), nn.ELU())] + \
            [nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ELU()) for _ in range(self.num_layers_h2z - 1)])
        )
        self.prior_logits = nn.Linear(self.hidden_dim, self.z_dim * self.num_classes)
        
        self.posterior_hidden = nn.Sequential(
            *([nn.Sequential(nn.Linear(self.h_dim + self.emb_dim, self.hidden_dim), nn.ELU())] + \
            [nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ELU()) for _ in range(self.num_layers_h2z - 1)])
        )
        self.posterior_logits = nn.Linear(self.hidden_dim, self.z_dim * self.num_classes)
    
    def recurrent(self, z, action, h):
        hidden = self.za2hidden(torch.concat([z, action], dim=1))
        next_h = self.transition(hidden, h)
        return next_h
    
    def prior(self, h, detach=False):
        hidden = self.prior_hidden(h)
        logits = self.prior_logits(hidden)
        logits = logits.reshape(logits.shape[0], self.z_dim, self.num_classes)
        prior = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        if detach:
            detach_prior = Independent(OneHotCategoricalStraightThrough(logits=logits.detach()), 1)
            return prior, detach_prior
        return prior
    
    def posterior(self, h, emb, detach=False):
        hidden = self.posterior_hidden(torch.concat([h, emb], dim=1))
        logits = self.posterior_logits(hidden)
        logits = logits.reshape(logits.shape[0], self.z_dim, self.num_classes)
        posterior = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        if detach:
            detach_posterior = Independent(OneHotCategoricalStraightThrough(logits=logits.detach()), 1)
            return posterior, detach_posterior
        return posterior


class ConvEncoder(nn.Module):
    def __init__(self, input_size, emb_dim):
        super(ConvEncoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([32, input_size // 2, input_size // 2]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([64, input_size // 4, input_size // 4]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([128, input_size // 8, input_size // 8]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([256, input_size // 16, input_size // 16]),
            nn.ReLU(),
        )

        self.fc = nn.Linear((input_size // 16) ** 2 * 256 , emb_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, img_size, z_dim, num_classes, h_dim):
        super(ConvDecoder, self).__init__()

        self.img_size = img_size
        self.fc = nn.Sequential(
            nn.Linear(z_dim * num_classes + h_dim, (img_size // 16) ** 2 * 256),
            nn.LayerNorm([(img_size // 16) ** 2 * 256,]),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LayerNorm([128, img_size // 8, img_size // 8]),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LayerNorm([64, img_size // 4, img_size // 4]),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LayerNorm([32, img_size // 2, img_size // 2]),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, z, h):
        x = torch.concat([z, h], dim=1)
        out = self.fc(x)
        out = out.reshape(out.shape[0], 256, self.img_size // 16, self.img_size // 16)
        out = self.net(out)
        dist = Independent(Normal(out, 1), 3)
        return dist


class Discount(nn.Module):
    def __init__(self, z_dim, num_classes, h_dim, hidden_dim=256):
        super(Discount, self).__init__()
        
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(z_dim * num_classes + h_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, z, h):
        logits = self.net(torch.concat([z, h], dim=1))
        dist = Independent(Bernoulli(logits=logits), 1)
        return dist


class ExprolerStatePredictor(nn.Module):
    def __init__(self, z_dim, num_classes, h_dim, target_dim, min_std, hidden_dim=256):
        super(ExprolerStatePredictor, self).__init__()
        
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.min_std = min_std
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        
        self.net = nn.Sequential(
            nn.Linear(z_dim * num_classes + h_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_fc = nn.Linear(hidden_dim, target_dim)
        self.std_fc = nn.Linear(hidden_dim, target_dim)
    
    def forward(self, z, h):
        h = self.net(torch.concat([z, h], dim=1))
        mean = self.mean_fc(h)
        std = self.std_fc(h) + self.min_std
        return Independent(Normal(mean, std), 1)


class ExplorerActor(nn.Module):
    def __init__(self, action_dim, z_dim, num_classes, h_dim, hidden_dim=256, min_std=0.1):
        super(ExplorerActor, self).__init__()
        
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std
        
        self.net = nn.Sequential(
            nn.Linear(z_dim * num_classes + h_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.std_fc = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, z, h, train=True):
        h = self.net(torch.concat([z, h], dim=1))
        mean = F.tanh(self.mean_fc(h))
        std = 2 * F.sigmoid(self.std_fc(h) / 2) + self.min_std
        dist = Independent(TruncatedNormal(mean, std, -1, 1), 1)
        if train:
            action = dist.rsample()
            log_prob = dist.log_prob(action.detach())
            entropy = dist.entropy()
            return action, log_prob, entropy
        else:
            action = dist.mean
            return action, None, None


class ExplorerCritic(nn.Module):
    def __init__(self, z_dim, num_classes, h_dim, hidden_dim=256):
        super(ExplorerCritic, self).__init__()
        
        self.z_dim = z_dim
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(z_dim * num_classes + h_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, z, h):
        return self.net(torch.concat([z, h], dim=1))


class State2Emb(nn.Module):
    def __init__(self, z_dim, num_classes, h_dim, emb_dim, min_std, hidden_dim=256):
        super(State2Emb, self).__init__()
        
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.min_std = min_std
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(z_dim * num_classes + h_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_fc = nn.Linear(hidden_dim, emb_dim)
        self.std_fc = nn.Linear(hidden_dim, emb_dim)
    
    def forward(self, z, h):
        h = self.net(torch.concat([z, h], dim=1))
        mean = self.mean_fc(h)
        std = self.std_fc(h) + self.min_std
        return Independent(Normal(mean, std), 1)


class AchieverDistanceEstimator(nn.Module):
    def __init__(self, emb_dim, hidden_dim=256):
        super(AchieverDistanceEstimator, self).__init__()
        
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        
        self.state2emb = nn
        
        self.current_fc = nn.Sequential(nn.Linear(emb_dim, hidden_dim),
                                        nn.GELU())
        self.goal_fc = nn.Sequential(nn.Linear(emb_dim, hidden_dim),
                                     nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, current_emb, goal_emb):
        cur_h = self.current_fc(current_emb)
        goal_h = self.goal_fc(goal_emb)
        return self.net(torch.concat([cur_h, goal_h], dim=1))


class AchieverCritic(nn.Module):
    def __init__(self, z_dim, num_classes, h_dim, emb_dim, hidden_dim=256):
        super(AchieverCritic, self).__init__()
        
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        
        self.state_fc = nn.Sequential(nn.Linear(z_dim * num_classes + h_dim, hidden_dim),
                                      nn.GELU())
        self.goal_fc = nn.Sequential(nn.Linear(emb_dim, hidden_dim),
                                     nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, z, h, goal_emb):
        state_h = self.state_fc(torch.concat([z, h], dim=1))
        goal_h = self.goal_fc(goal_emb)
        return self.net(torch.concat([state_h, goal_h], dim=1))


class AchieverActor(nn.Module):
    def __init__(self, action_dim, z_dim, num_classes, h_dim, emb_dim, hidden_dim=256, min_std=0.1):
        super(AchieverActor, self).__init__()
        
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std
        
        self.state_fc = nn.Sequential(nn.Linear(z_dim * num_classes + h_dim, hidden_dim),
                                      nn.GELU())
        self.goal_fc = nn.Sequential(nn.Linear(emb_dim, hidden_dim),
                                     nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.std_fc = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, z, h, goal_emb, train=True):
        state_h = self.state_fc(torch.concat([z, h], dim=1))
        goal_h = self.goal_fc(goal_emb)
        h = self.net(torch.concat([state_h, goal_h], dim=1))
        mean = F.tanh(self.mean_fc(h))
        std = 2 * F.sigmoid(self.std_fc(h) / 2) + self.min_std
        dist = Independent(TruncatedNormal(mean, std, -1, 1), 1)
        if train:
            action = dist.rsample()
            log_prob = dist.log_prob(action.detach())
            entropy = dist.entropy()
            return action, log_prob, entropy
        else:
            action = dist.mean
        return action, None, None
