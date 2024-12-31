from typing import Union, Literal
import functools
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from gymnasium import Env

from config import Config
from model.worldmodel import WorldModel
from model.explorer import Explorer
from model.explorer_reward import EmsembleReward, RSSMRndReward, FeedForwardRndReward
from model.achiever import Achiever
from model.achiever_reward import LatentDistanceReward
from replay_buffer import ReplayBuffer


class LEXA:
    def __init__(self, cfg: Config, env: Env):
        self.cfg = cfg
        self.env = env
        self.device = torch.device(self.cfg.device)
        
        self.world_model = WorldModel(
            img_size = cfg.env.img_size,
            emb_dim = cfg.model.world_model.emb_dim,
            action_dim = self.env.action_space.shape[0],
            z_dim = cfg.model.world_model.z_dim,
            num_classes = cfg.model.world_model.num_classes,
            h_dim = cfg.model.world_model.h_dim,
            hidden_dim = cfg.model.world_model.hidden_dim,
            num_layers_za2hidden = cfg.model.world_model.num_layers_za2hidden,
            num_layers_h2z = cfg.model.world_model.num_layers_h2z,
            mlp_hidden_dim = cfg.model.world_model.mlp_hidden_dim,
            min_std = cfg.model.world_model.min_std,
            kl_balance_alpha = cfg.model.world_model.kl_balance_alpha,
            kl_loss_scale = cfg.model.world_model.kl_loss_scale,
            device = self.device
        ).to(self.device)
        self.explorer_reward = EmsembleReward(
            z_dim = cfg.model.world_model.z_dim,
            num_classes = cfg.model.world_model.num_classes,
            h_dim = cfg.model.world_model.h_dim,
            min_std = cfg.model.world_model.min_std,
            mlp_hidden_dim = cfg.model.explorer.mlp_hidden_dim,
            device = self.device,
            num_emsembles = cfg.model.explorer.num_emsembles,
            offset = cfg.model.explorer.emsembles_offset,
            target_mode = cfg.model.explorer.emsembles_target_mode,
        ).to(self.device)

        # self.explorer_rnd_reward = RSSMRndReward(
        #     rnd_target= self.world_model.rssm_tartet,
        #     rnd_predictor= self.world_model.rssm_predictor,
        #     z_dim = cfg.model.world_model.z_dim,
        #     num_classes = cfg.model.world_model.num_classes,
        #     h_dim = cfg.model.world_model.h_dim,
        #     device = self.device,
        #     target_mode = cfg.model.explorer.emsembles_target_mode,
        # ).to(self.device)
        self.explorer_rnd_reward = FeedForwardRndReward(
            z_dim = cfg.model.world_model.z_dim,
            num_classes = cfg.model.world_model.num_classes,
            h_dim = cfg.model.world_model.h_dim,
            min_std = cfg.model.world_model.min_std,
            mlp_hidden_dim = cfg.model.explorer.mlp_hidden_dim,
        )

        self.explorer = Explorer(
            world_model = self.world_model,
            instrinsic_reward = self.explorer_reward,
            rnd_reward = self.explorer_rnd_reward,
            action_dim = self.env.action_space.shape[0],
            z_dim = cfg.model.world_model.z_dim,
            num_classes = cfg.model.world_model.num_classes,
            h_dim = cfg.model.world_model.h_dim,
            mlp_hidden_dim = cfg.model.explorer.mlp_hidden_dim,
            min_std = cfg.model.explorer.min_std,
            num_emsembles = cfg.model.explorer.num_emsembles,
            emsembles_offset = cfg.model.explorer.emsembles_offset,
            emsembles_target_mode = cfg.model.explorer.emsembles_target_mode,
            discount = cfg.model.explorer.discount,
            lambda_ = cfg.model.explorer.lambda_,
            actor_entropy_scale = cfg.model.explorer.actor_entropy_scale,
            device = self.device
        ).to(self.device)
        self.achiever_reward = LatentDistanceReward(
            z_dim = cfg.model.world_model.z_dim,
            num_classes = cfg.model.world_model.num_classes,
            h_dim = cfg.model.world_model.h_dim,
            emb_dim = cfg.model.world_model.emb_dim,
            mlp_hidden_dim = cfg.model.achiever.mlp_hidden_dim,
            min_std = cfg.model.achiever.min_std,
            device = self.device
        ).to(self.device)
        self.achiever = Achiever(
            world_model = self.world_model,
            instrinsic_reward = self.achiever_reward,
            action_dim = self.env.action_space.shape[0],
            z_dim = cfg.model.world_model.z_dim,
            num_classes = cfg.model.world_model.num_classes,
            h_dim = cfg.model.world_model.h_dim,
            emb_dim = cfg.model.world_model.emb_dim,
            mlp_hidden_dim = cfg.model.achiever.mlp_hidden_dim,
            min_std = cfg.model.achiever.min_std,
            discount = cfg.model.achiever.discount,
            lambda_ = cfg.model.achiever.lambda_,
            actor_entropy_scale = cfg.model.achiever.actor_entropy_scale,
            device = self.device
        ).to(self.device)
        
        self.wm_opt = optim.Adam(self.world_model.parameters(),
                                 lr = cfg.learning.world_model_lr,
                                 eps = cfg.learning.epsilon,
                                 weight_decay = cfg.learning.weight_decay)
        self.exp_reward_opt = optim.Adam(list(self.explorer_reward.parameters()) + list(self.explorer_rnd_reward.parameters()),
                                         lr = cfg.learning.world_model_lr,
                                         eps = cfg.learning.epsilon,
                                         weight_decay = cfg.learning.weight_decay)
        self.exp_actor_opt = optim.Adam(self.explorer.actor.parameters(),
                                        lr = cfg.learning.explorer_actor_lr,
                                        eps = cfg.learning.epsilon,
                                        weight_decay = cfg.learning.weight_decay)
        self.exp_critic_opt = optim.Adam(self.explorer.critic.parameters(),
                                         lr = cfg.learning.explorer_critic_lr,
                                         eps = cfg.learning.epsilon,
                                         weight_decay = cfg.learning.weight_decay)
        self.ach_reward_opt = optim.Adam(self.achiever_reward.parameters(),
                                        lr = cfg.learning.achiever_critic_lr,
                                        eps = cfg.learning.epsilon,
                                        weight_decay = cfg.learning.weight_decay)
        self.ach_actor_opt = optim.Adam(self.achiever.actor.parameters(),
                                        lr = cfg.learning.achiever_actor_lr,
                                        eps = cfg.learning.epsilon,
                                        weight_decay = cfg.learning.weight_decay)
        self.ach_critic_opt = optim.Adam(self.achiever.critic.parameters(),
                                         lr = cfg.learning.achiever_critic_lr,
                                         eps = cfg.learning.epsilon,
                                         weight_decay = cfg.learning.weight_decay)
        
        self.agent = Agent(self.world_model, self.explorer, self.achiever, self.device)
    
    def train(self, observations, actions):
        observations = torch.from_numpy(observations).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        
        wm_loss, (zs, hs), wm_metrics = self.world_model.train(observations, actions)
        emsemble_loss, emsemble_metrics = self.explorer_reward.train(zs, hs)
        rnd_loss, rnd_metrics = self.explorer_rnd_reward.train(zs, hs)
        self.wm_opt.zero_grad(True)
        wm_loss.backward()
        clip_grad_norm_(self.world_model.parameters(), self.cfg.learning.grad_clip)
        self.wm_opt.step()
        self.exp_reward_opt.zero_grad(True)
        emsemble_loss.backward()
        rnd_loss.backward()
        clip_grad_norm_(self.explorer_reward.parameters(), self.cfg.learning.grad_clip)
        self.exp_reward_opt.step()
        
        zs = zs.view(-1, self.cfg.model.world_model.z_dim * self.cfg.model.world_model.num_classes)
        hs = hs.view(-1, self.cfg.model.world_model.h_dim)
        
        exp_actor_loss, axp_critic_loss, exp_metrics = self.explorer.train(zs, hs,observations, self.cfg.data.imagination_horizon)
        self.exp_actor_opt.zero_grad(True)
        exp_actor_loss.backward()
        clip_grad_norm_(self.explorer.actor.parameters(), self.cfg.learning.grad_clip)
        self.exp_actor_opt.step()
        self.exp_critic_opt.zero_grad(True)
        axp_critic_loss.backward()
        clip_grad_norm_(self.explorer.critic.parameters(), self.cfg.learning.grad_clip)
        self.exp_critic_opt.step()
        
        ach_actor_loss, ach_critic_loss, de_loss, ach_metrics = self.achiever.train(zs, hs, observations,
                                                                                    self.cfg.data.imagination_horizon,
                                                                                    self.cfg.model.achiever.num_positives,
                                                                                    self.cfg.model.achiever.neg_sampling_factor,
                                                                                    self.cfg.data.seq_length)
        self.ach_actor_opt.zero_grad(True)
        ach_actor_loss.backward()
        clip_grad_norm_(self.achiever.actor.parameters(), self.cfg.learning.grad_clip)
        self.ach_actor_opt.step()
        self.ach_critic_opt.zero_grad(True)
        ach_critic_loss.backward()
        clip_grad_norm_(self.achiever.critic.parameters(), self.cfg.learning.grad_clip)
        self.ach_critic_opt.step()
        self.ach_reward_opt.zero_grad(True)
        de_loss.backward()
        clip_grad_norm_(self.achiever_reward.parameters(), self.cfg.learning.grad_clip)
        self.ach_reward_opt.step()
        
        return wm_metrics | emsemble_metrics | rnd_metrics | exp_metrics | ach_metrics
    
    @staticmethod
    def load(checkpoint):
        cfg = checkpoint['config']
        env = checkpoint['env']
        output = LEXA(cfg, env)
        output.world_model.load_state_dict(checkpoint['world_model'])
        output.explorer_reward.load_state_dict(checkpoint['exp_reward'])
        output.explorer.load_state_dict(checkpoint['explorer'])
        output.achiever_reward.load_state_dict(checkpoint['ach_reward'])
        output.achiever.load_state_dict(checkpoint['achiever'])
        output.wm_opt.load_state_dict(checkpoint['wm_opt'])
        output.exp_reward_opt.load_state_dict(checkpoint['exp_reward_opt'])
        output.exp_actor_opt.load_state_dict(checkpoint['exp_actor_opt'])
        output.exp_critic_opt.load_state_dict(checkpoint['exp_critic_opt'])
        output.ach_reward_opt.load_state_dict(checkpoint['ach_reward_opt'])
        output.ach_actor_opt.load_state_dict(checkpoint['ach_actor_opt'])
        output.ach_critic_opt.load_state_dict(checkpoint['ach_critic_opt'])
        return output
    
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'world_model': self.world_model.state_dict(),
                'exp_reward': self.explorer_reward.state_dict(),
                'explorer': self.explorer.state_dict(),
                'ach_reward': self.achiever_reward.state_dict(),
                'achiever': self.achiever.state_dict(),
                'wm_opt': self.wm_opt.state_dict(),
                'exp_reward_opt': self.exp_reward_opt.state_dict(),
                'exp_actor_opt': self.exp_actor_opt.state_dict(),
                'exp_critic_opt': self.exp_critic_opt.state_dict(),
                'ach_reward_opt': self.ach_reward_opt.state_dict(),
                'ach_actor_opt': self.ach_actor_opt.state_dict(),
                'ach_critic_opt': self.ach_critic_opt.state_dict(),
                'config': self.cfg,
                'env': self.env,
            },
            path
        )


class Agent:
    def __init__(self, world_model: WorldModel, explorer: Explorer, achiever: Achiever, device: torch.device):
        self.world_model = world_model
        self.explorer = explorer
        self.achiever = achiever
        self.device = device
        
        self.h = torch.zeros(1, self.world_model.h_dim, device=self.device)
    
    @torch.no_grad()
    def __call__(self, observation, mode: Literal['explorer', 'achiever'], goal=None, train=True):
        observation = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        
        if mode == 'explorer':
            policy = self.explorer.actor
        elif mode == 'achiever':
            policy = self.achiever.actor
            assert goal is not None, 'goal must be provided in achiever mode'
            goal = torch.from_numpy(goal).unsqueeze(0).to(self.device)
            goal_emb = self.world_model.encoder(goal)
            policy = functools.partial(policy, goal_emb=goal_emb)
        
        obs_emb = self.world_model.encoder(observation)
        z_posterior = self.world_model.rssm.posterior(self.h, obs_emb)
        z = z_posterior.sample().flatten(1)
        action, _, _ = policy(z, self.h, train=train)
        
        self.h = self.world_model.rssm.recurrent(z, action, self.h)
        
        return action.squeeze().cpu().numpy()
    
    def reset(self):
        self.h = torch.zeros(1, self.world_model.h_dim, device=self.device)
