import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al. 2022).
    Includes safety clamping and output clipping to prevent signal explosion.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        # Stores mean/std for denormalization
        self.mean = None
        self.std = None

    def forward(self, x, mode: str = "norm"):
        if mode == "norm":
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(f"mode must be 'norm' or 'denorm', got '{mode}'")

    def _normalize(self, x):
        """
        Normalize input and store mean/std.
        x: (batch, channels, seq_len)
        """
        self.mean = x.mean(dim=-1, keepdim=True).detach()
        self.std = x.std(dim=-1, keepdim=True).detach()

        # SAFETY: Clamp std to prevent division by zero or tiny values
        self.std = torch.clamp(self.std, min=1e-6)

        x_norm = (x - self.mean) / self.std

        if self.affine:
            x_norm = x_norm * self.affine_weight.view(1, -1, 1) + self.affine_bias.view(
                1, -1, 1
            )

        # SAFETY: Clip extreme values to prevent physical saturation of the model
        x_norm = torch.clamp(x_norm, min=-10.0, max=10.0)

        return x_norm

    def _denormalize(self, x):
        """
        Denormalize predictions back to original scale.
        x: (batch, channels, pred_len)
        """
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "RevIN: must call forward(x, 'norm') before forward(x, 'denorm')"
            )

        if self.affine:
            x = (x - self.affine_bias.view(1, -1, 1)) / self.affine_weight.view(
                1, -1, 1
            )

        return x * self.std + self.mean


class Transpose(nn.Module):
    """Helper for BatchNorm compatibility."""

    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
