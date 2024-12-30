from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from einops import rearrange

from .network import ExplorerActor, ExplorerCritic
from .worldmodel import WorldModel
from .explorer_reward import EmsembleReward
from .utils import compute_lambda_target
from torch import optim

from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader


class Explorer(nn.Module):
    def __init__(self,
                 world_model: WorldModel,
                 instrinsic_reward,
                 action_dim,
                 z_dim,
                 num_classes,
                 h_dim,
                 mlp_hidden_dim,
                 min_std,
                 num_emsembles,
                 emsembles_offset,
                 emsembles_target_mode,
                 discount,
                 lambda_,
                 actor_entropy_scale,
                 device):
        super(Explorer, self).__init__()
        
        self.world_model = world_model
        
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.h_dim = h_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.min_std = min_std
        self.num_emsembles = num_emsembles
        self.emsembles_offset = emsembles_offset
        self.emsembles_target_mode = emsembles_target_mode
        self.discount = discount
        self.lambda_ = lambda_
        self.actor_entropy_scale = actor_entropy_scale
        self.device = device
        self.lr= 0.001 #適当
        self.opt = optim.Adam(lr=self.lr, params=self.world_model.encoder_rnd_predictor.parameters())
        
        self.actor = ExplorerActor(
            action_dim = action_dim,
            z_dim = z_dim,
            num_classes = num_classes,
            h_dim = h_dim,
            hidden_dim = mlp_hidden_dim,
            min_std = min_std
        )
        self.critic = ExplorerCritic(
            z_dim = z_dim,
            num_classes = num_classes,
            h_dim = h_dim,
            hidden_dim = mlp_hidden_dim
        )
        self.target_critic = ExplorerCritic(
            z_dim = z_dim,
            num_classes = num_classes,
            h_dim = h_dim,
            hidden_dim = mlp_hidden_dim
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.instrinsic_reward = instrinsic_reward
    
    def train(self, init_zs: torch.Tensor, init_hs: torch.Tensor, observations: torch.Tensor,horison_length):
        zs = init_zs.detach() # (batch_size * seq_length, z_dim * num_classes)
        hs = init_hs.detach() # (batch_size * seq_length, h_dim)
        
        imagined_zs = torch.empty(horison_length, *init_zs.shape, device=self.device)
        imagined_hs = torch.empty(horison_length, *init_hs.shape, device=self.device)
        imagined_action_log_probs = torch.empty(horison_length, init_zs.shape[0], device=self.device)
        imagined_action_entropys = torch.empty(horison_length, init_zs.shape[0], device=self.device)
        
        for t in range(horison_length):
            actions, action_log_probs, action_entropys = self.actor(zs.detach(), hs.detach())
            
            with torch.no_grad():
                hs, zs = self.world_model.imagine(actions, zs, hs)
            
            imagined_hs[t] = hs.detach()
            imagined_zs[t] = zs.detach()
            imagined_action_log_probs[t] = action_log_probs
            imagined_action_entropys[t] = action_entropys
        
        flatten_hs = imagined_hs.view(-1, self.h_dim).detach() # (horison_length * batch_size * seq_length, h_dim)
        flatten_zs = imagined_zs.view(-1, self.z_dim * self.num_classes).detach() # (horison_length * batch_size * seq_length, z_dim * num_classes)
        
        # ざっくりRND reward実装
        # todo 
            # lossの計算の前にこんな感じでshapeを変えた方が良い。
                # beta_t = self.beta * np.power(1. - self.kappa, time_steps)
                # n_steps = rollouts['observations'].shape[0]
                # n_envs = rollouts['observations'].shape[1]
                # intrinsic_rewards = np.zeros(shape=(n_steps, n_envs, 1))
                # with torch.no_grad():
                #     for idx in range(n_envs):
                #         src_feats = self.predictor(obs_tensor[:, idx])
                #         tgt_feats = self.target(obs_tensor[:, idx])
                #         dist = F.mse_loss(src_feats, tgt_feats, reduction='none').mean(dim=1)
                #         dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-11)
                #         intrinsic_rewards[:-1, idx, 0] = dist[1:].cpu().numpy()
            # rnd_rewardsのshapeを調整して、本家のrewardと足し合わせる
            # encoderとは別に、RND用のencoderを二つ用意しているが、これでいいのか？を検討

        # RNDの報酬の計算
        with torch.no_grad():
            embs_encoder_rnd_target = self.world_model.encoder_rnd_target(rearrange(observations, 't b c h w -> (t b) c h w'))
            embs_encoder_rnd_predictor = self.world_model.encoder_rnd_predictor(rearrange(observations, 't b c h w -> (t b) c h w'))            
            rnd_rewards = F.mse_loss(embs_encoder_rnd_target -embs_encoder_rnd_predictor)

        dataset = TensorDataset(observations)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        # RNDのネットワークの更新。
        # encoder_rnd_targetは学習しない。観測されたobservationについては、学習によりencoder_rnd_predictorがencoder_rnd_targetに近づくため上記の報酬が小さくなる
        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            src_feats = self.world_model.encoder_rnd_target(rearrange(batch_obs, 't b c h w -> (t b) c h w'))
            tgt_feats = self.world_model.encoder_rnd_target(rearrange(batch_obs, 't b c h w -> (t b) c h w'))
            loss = F.mse_loss(src_feats, tgt_feats)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        with torch.no_grad():
            rewards = self.instrinsic_reward.compute_reward(flatten_zs, flatten_hs).view(horison_length, -1) # (horison_length, batch_size * seq_length)
            # rewards = rewards + rnd_rewards #こんな感じでもともとのrewardに、RNDの報酬を足したい
            target_values = self.target_critic(flatten_zs, flatten_hs).view(horison_length, -1) # (horison_length, batch_size * seq_length)
        
        lambda_target = compute_lambda_target(rewards, self.discount, target_values, self.lambda_)
        
        objective = imagined_action_log_probs * ((lambda_target - target_values).detach())
        actor_loss = -torch.sum(torch.mean(objective + self.actor_entropy_scale * imagined_action_entropys, dim=1))
        
        value_mean = self.critic(flatten_zs.detach(), flatten_hs.detach()).view(horison_length, -1)
        value_dist = td.Independent(td.Normal(value_mean, 1),  1)
        critic_loss = -torch.mean(value_dist.log_prob(lambda_target.detach()).unsqueeze(-1))
        
        return actor_loss, critic_loss, OrderedDict(exp_actor_loss=actor_loss.item(), exp_critic_loss=critic_loss.item())
    
    def update_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
