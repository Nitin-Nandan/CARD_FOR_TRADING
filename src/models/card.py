"""
TRUE CARD: Channel Aligned Robust Blend Transformer
Implementation from ICLR 2024 Paper

Reference: "CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting"
Authors: Wang Xue, Tian Zhou, et al.
"""

import torch
import torch.nn as nn
from .layers import RevIN
from .attention import CARDAttention


class CARD(nn.Module):
    """
    TRUE CARD Model (ICLR 2024) — Clean rewrite for close price prediction.

    Architecture:
        Input (B, C, L)
        → RevIN normalize (per-channel, over time)
        → Patch tokenization + positional embedding + CLS token
        → [Channel Attention + Token Attention] × e_layers
        → MLP decoder  →  (B, C, pred_len)
        → RevIN denormalize (back to price scale)
        → Extract close channel  →  (B, pred_len)

    Args:
        config: CARDModelConfig with all hyperparameters
    """

    def __init__(self, config):
        super(CARD, self).__init__()

        self.patch_len = config.patch_len
        self.stride = config.stride
        self.d_model = config.d_model
        self.close_channel_idx = getattr(
            config, "close_channel_idx", 3
        )  # default: channel 3

        # Number of patches: N = floor((L-P)/S + 1)
        patch_num = int((config.seq_len - self.patch_len) / self.stride + 1)
        self.patch_num = patch_num
        self.total_token_number = patch_num + 1  # +1 for CLS
        config.total_token_number = self.total_token_number

        # ------------------------------------------------------------
        # RevIN
        # ------------------------------------------------------------
        self.revin = RevIN(num_features=config.enc_in, eps=1e-5, affine=False)

        # ------------------------------------------------------------
        # TOKENIZATION (Section 3.1)
        # ------------------------------------------------------------
        self.W_input_projection = nn.Linear(self.patch_len, config.d_model)
        self.W_pos_embed = nn.Parameter(torch.randn(patch_num, config.d_model) * 1e-2)
        self.cls = nn.Parameter(torch.randn(1, config.d_model) * 1e-2)
        self.input_dropout = nn.Dropout(config.dropout)

        # ------------------------------------------------------------
        # DUAL ATTENTION ENCODER (Sections 3.2-3.3)
        # ------------------------------------------------------------
        self.Attentions_over_token = nn.ModuleList(
            [CARDAttention(config, over_channel=False) for _ in range(config.e_layers)]
        )
        self.Attentions_over_channel = nn.ModuleList(
            [CARDAttention(config, over_channel=True) for _ in range(config.e_layers)]
        )
        self.Attentions_mlp = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model) for _ in range(config.e_layers)]
        )
        self.Attentions_dropout = nn.ModuleList(
            [nn.Dropout(config.dropout) for _ in range(config.e_layers)]
        )

        # Import Transpose here or use a dummy if not needed directly in the main model.
        # Actually, Transpose was used in self.Attentions_norm.
        from .layers import Transpose

        self.Attentions_norm = nn.ModuleList(
            [
                nn.Sequential(
                    Transpose(1, 2),
                    nn.BatchNorm1d(config.d_model, momentum=config.momentum),
                    Transpose(1, 2),
                )
                for _ in range(config.e_layers)
            ]
        )

        # ------------------------------------------------------------
        # MLP DECODER: (patch_num+1)*d_model → pred_len
        # ------------------------------------------------------------
        self.W_out = nn.Linear((patch_num + 1) * config.d_model, config.pred_len)

    def forward(self, z):
        """
        Args:
            z: (batch, channels, seq_len)   e.g. (B, 18, 60)

        Returns:
            pred_returns: (batch, pred_len) — predicted returns
        """
        b, c, s = z.shape

        # 1. RevIN NORMALIZE
        z = self.revin(z, mode="norm")

        # 2. PATCH TOKENIZATION
        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z_embed = self.input_dropout(self.W_input_projection(zcube)) + self.W_pos_embed

        # Prepend CLS token
        cls_token = self.cls.expand(b, c, 1, -1)
        z_embed = torch.cat((cls_token, z_embed), dim=2)

        # 3. DUAL ATTENTION ENCODER
        inputs = z_embed
        b2, c2, t, h = inputs.shape

        for a_token, a_channel, mlp, drop, norm in zip(
            self.Attentions_over_token,
            self.Attentions_over_channel,
            self.Attentions_mlp,
            self.Attentions_dropout,
            self.Attentions_norm,
        ):
            output_channel = a_channel(inputs.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            output_token = a_token(output_channel)
            outputs = drop(mlp(output_channel + output_token)) + inputs
            outputs = norm(outputs.reshape(b2 * c2, t, -1)).reshape(b2, c2, t, -1)
            inputs = outputs

        # 4. MLP DECODER
        z_flat = outputs.reshape(b, c, -1)
        pred_returns = self.W_out(z_flat)

        # Extract close channel
        pred_returns = pred_returns[:, self.close_channel_idx, :]

        # Limit predictions to reasonable return values
        pred_returns = torch.clamp(pred_returns, min=-0.2, max=0.2)

        return pred_returns
