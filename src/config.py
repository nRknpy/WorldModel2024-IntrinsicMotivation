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


@dataclass
class AchieverConfig:
    mlp_hidden_dim: int = 400
    min_std: float = 0.1
    discount: float = 0.99
    lambda_: float = 0.95
    actor_entropy_scale: float = 1e-4


@dataclass
class LEXAConfig:
    world_model: WorldModelConfig = WorldModelConfig()
    explorer: ExplorerConfig = ExplorerConfig()
    achiever: AchieverConfig = AchieverConfig()
    