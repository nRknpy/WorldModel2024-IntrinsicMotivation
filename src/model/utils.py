import numpy as np
from PIL import Image
import torch
import torch.distributions as td
from torch.distributions.utils import _standard_normal
from einops import rearrange


class TruncatedNormal(td.Normal):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, low: float = -1.0, high: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip: float | None = None, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)

        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def compute_lambda_target(rewards: torch.Tensor, discount: float, values: torch.Tensor, lambda_: float):
    V_lambda = torch.zeros_like(rewards)

    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            V_lambda[t] = rewards[t] + discount * values[t]
        else:
            V_lambda[t] = rewards[t] + discount * ((1-lambda_) * values[t+1] + lambda_ * V_lambda[t+1])

    return V_lambda
