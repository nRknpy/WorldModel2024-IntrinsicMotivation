from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    emb_dim: int = 1024
    z_dim: int = 32
    num_classes: int = 32
    h_dim: int = 600
    hidden_dim: int = 600
    num_layers_za2hidden: int = 1
    num_layers_h2z: int = 1
    mlp_hidden_dim: int = 400
    min_std: float = 0.1
    kl_balance_alpha: float = 0.8
    kl_loss_scale: float = 0.1


@dataclass
class ExplorerConfig:
    num_emsembles: int = 10
    emsembles_offset: int = 1
    emsembles_target_mode: str = 'z'
    mlp_hidden_dim: int = 400
    min_std: float = 0.1
    discount: float = 0.99
    lambda_: float = 0.95
    actor_entropy_scale: float = 1e-4
    slow_critic_update: int = 100


@dataclass
class AchieverConfig:
    num_positives: int = 256
    neg_sampling_factor: int = 0.1
    mlp_hidden_dim: int = 400
    min_std: float = 0.1
    discount: float = 0.99
    lambda_: float = 0.95
    actor_entropy_scale: float = 1e-4
    slow_critic_update: int = 100


@dataclass
class LEXAModelConfig:
    world_model: WorldModelConfig = WorldModelConfig()
    explorer: ExplorerConfig = ExplorerConfig()
    achiever: AchieverConfig = AchieverConfig()


@dataclass
class DataConfig:
    buffer_size: int = 2e6
    batch_size: int = 50
    seq_length: int = 50
    imagination_horizon: int = 15


@dataclass
class LearningConfig:
    seed_steps: int = 0
    num_steps: int = 2e6
    expl_episode_freq: int = 2
    world_model_lr: float = 2e-4
    explorer_actor_lr: float = 4e-5
    explorer_critic_lr: float = 1e-4
    achiever_actor_lr: float = 4e-5
    achiever_critic_lr: float = 1e-4
    epsilon: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip: float = 1000
    update_freq: int = 5
    eval_episode_freq: int = 100
    model_save_episode_freq: int = 1000


@dataclass
class EnvConfig:
    task: str = 'PointMaze'
    img_size: int = 64
    action_repeat: int = 5
    time_limit: int = 1000


@dataclass
class WandbConfig:
    logging: bool = False
    name: str = 'lexa'
    group: str = ''
    project: str = 'LEXA'


@dataclass
class Config:
    model: LEXAModelConfig = LEXAModelConfig()
    env: EnvConfig = EnvConfig()
    data: DataConfig = DataConfig()
    learning: LearningConfig = LearningConfig()
    wandb: WandbConfig = WandbConfig()
    device: str = 'cuda'
    seed: int = 0
