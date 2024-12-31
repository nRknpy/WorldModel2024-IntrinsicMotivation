from typing import Literal
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .network import ExprolerStatePredictor
from torch.distributions import Normal, kl_divergence
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader



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
        reward = torch.sum(var, dim=1)
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
    

class RndReward(nn.Module):
    def __init__(self,
                 rnd_target,
                 rnd_predictor,
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
        self.rnd_target=rnd_target
        self.rnd_predictor=rnd_predictor
        # self.z_dim = z_dim
        # self.num_classes = num_classes
        # self.h_dim = h_dim
        # self.min_std = min_std
        # self.num_emsembles = num_emsembles
        # self.offset = offset
        # self.target_mode = target_mode
        # self.mlp_hidden_dim = mlp_hidden_dim
        self.device = device
        self.lr= 0.001 #適当
        self.opt = optim.Adam(lr=self.lr, params=self.rnd_predictor.parameters())
        
        if target_mode == 'z':
            self.target_dim = z_dim * num_classes
        elif target_mode == 'h':
            self.target_dim = h_dim
        elif target_mode == 'zh':
            self.target_dim = z_dim * num_classes + h_dim

    def imagine(self, network, action, z, h):
        next_h = network.recurrent(z, action, h)
        next_prior = network.prior(next_h)
        next_z = next_prior.rsample().flatten(1)
        return next_h, next_z    

    # 特徴ベクトルを分布に変換する関数
    def to_distribution(features):
        """
        特徴ベクトルを分布の平均と分散に変換
        Args:
            features: Tensor (batch_size * seq_length, feature_dim)
        Returns:
            mean: 平均
            std: 標準偏差
        """
        # TODO  バッチごとに平均と分散を計算するような処理に変える
        mean, log_std = features.chunk(2, dim=-1)  # 分布のパラメータに分割
        std = torch.exp(log_std)  # 標準偏差を計算 (expで戻す)
        return mean, std

    # KL ダイバージェンスを計算する関数
    def compute_kl_divergence(self,rnd_target_feature, rnd_predictor_feature, h_dim):
        """
        rnd_target_next_h と rnd_predictor_next_h の間の KL ダイバージェンスを計算
        Args:
            rnd_target: Tensor (batch_size * seq_length, h_dim)
            rnd_predictor: Tensor (batch_size * seq_length, h_dim)
            h_dim: 次元数
        Returns:
            kl_loss: KL ダイバージェンスの損失
        """
        # zやhを分布に変換
        # TODO target_modeなどを使って、zとhの両方に対応できるように書き換える

        target_mean, target_std = self.to_distribution(rnd_target_feature)
        predictor_mean, predictor_std = self.to_distribution(rnd_predictor_feature)

        # 分布を定義
        target_distribution = Normal(target_mean, target_std)
        predictor_distribution = Normal(predictor_mean, predictor_std)

        # KL ダイバージェンスを計算
        kl_loss = kl_divergence(target_distribution, predictor_distribution).mean()  # 平均を取る
        return kl_loss

    #報酬を計算
    def compute_rnd_reward(self, imagined_zs, imagined_hs):
        reward=0
        with torch.no_grad():
            for i in range(len(imagined_zs)):
                rnd_target_next_h, rnd_target_next_z = self.imagine(self.rnd_target,imagined_zs[i],imagined_hs[i])
                rnd_predictor_next_h, rnd_predictor_next_z = self.imagine(self.rnd_predictor,imagined_zs[i],imagined_hs[i])
                
                # KL ダイバージェンスを計算
                kl_loss_h = self.compute_kl_divergence(rnd_target_next_h, rnd_predictor_next_h)
                kl_loss_z = self.compute_kl_divergence(rnd_target_next_z, rnd_predictor_next_z)
                reward += kl_loss_h +kl_loss_z
            reward /= len(imagined_zs)*2
        
        self.update(imagined_zs, imagined_hs)

        return torch.tensor(reward)
 

    def update(self, imagined_zs, imagined_hs):

        dataset = TensorDataset(imagined_zs,imagined_hs)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        # RNDのネットワークの更新。targetは更新しない。
        for idx, (z,h) in enumerate(loader):
            rnd_target_next_h, rnd_target_next_z = self.imagine(self.rnd_target,z,h)
            rnd_predictor_next_h, rnd_predictor_next_z = self.imagine(self.rnd_predictor,z,h)
            loss = self.compute_kl_divergence(rnd_target_next_h, rnd_predictor_next_h)+self.compute_kl_divergence(rnd_target_next_z, rnd_predictor_next_z)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

