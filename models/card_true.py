"""
TRUE CARD: Channel Aligned Robust Blend Transformer
Implementation from ICLR 2024 Paper

Paper: "CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting"
Authors: Wang Xue, Tian Zhou, et al.
Published: ICLR 2024

PHASE 0 CLEAN REWRITE:
- Predicts future CLOSE PRICES (not returns)
- RevIN normalization + denormalization ENABLED (same-domain prediction)
- Single-output decoder (no volatility head)
- Both attention scaling bugs fixed (** 0.5 not ** -0.5)
- Single SignalDecayMAE loss (no scale_factor hack)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


# ============================================================================
# 1. RevIN (Reversible Instance Normalization)
# ============================================================================

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al. 2022)

    Normalizes each channel independently over the time dimension.
    Stores mean/std for denormalization of predictions.

    Args:
        num_features: Number of input channels (C)
        eps: Small constant for numerical stability
        affine: Learnable affine parameters (disabled by default)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        # Will be populated during forward (norm) and used in denorm
        self.mean = None
        self.std = None

    def forward(self, x, mode: str = 'norm'):
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"mode must be 'norm' or 'denorm', got '{mode}'")

    def _normalize(self, x):
        """
        Normalize input.
        x: (batch, seq_len, num_features)  OR  (batch, num_features, seq_len)

        We assume x is (batch, num_features, seq_len) — CARD convention.
        Normalize over seq_len dimension (dim=-1).
        """
        # x: (batch, channels, seq_len)
        self.mean = x.mean(dim=-1, keepdim=True).detach()   # (batch, channels, 1)
        self.std  = x.std(dim=-1, keepdim=True).detach() + self.eps  # (batch, channels, 1)

        x_norm = (x - self.mean) / self.std

        if self.affine:
            x_norm = x_norm * self.affine_weight.view(1, -1, 1) + self.affine_bias.view(1, -1, 1)

        return x_norm

    def _denormalize(self, x):
        """
        Denormalize predictions back to original scale.
        x: (batch, channels, pred_len)
        Uses the mean/std stored during the most recent _normalize call.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("RevIN: must call forward(x, 'norm') before forward(x, 'denorm')")

        if self.affine:
            x = (x - self.affine_bias.view(1, -1, 1)) / self.affine_weight.view(1, -1, 1)

        return x * self.std + self.mean


# ============================================================================
# 2. Helper
# ============================================================================

class Transpose(nn.Module):
    """Helper for BatchNorm compatibility."""
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


# ============================================================================
# 3. CARD Attention Module
# ============================================================================

class CARDAttention(nn.Module):
    """
    CARD Attention from ICLR 2024 paper (Section 3.2-3.4).

    Features:
    1. EMA-smoothed Q and K with FIXED alpha (Equations 3-4)
    2. Token attention: attention over temporal tokens
    3. Hidden-dim attention: attention over head_dim (Equation 4)
    4. Dynamic projection for channel attention (Equations 6-7)
    5. Token blend at HEAD level (Section 3.4)
    6. BatchNorm (not LayerNorm)

    Args:
        config: model config
        over_channel: True = channel attention, False = token attention
    """

    def __init__(self, config, over_channel: bool = False):
        super(CARDAttention, self).__init__()

        self.over_channel = over_channel
        self.n_heads = config.n_heads
        self.merge_size = config.merge_size
        self.c_in = config.enc_in
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        # QKV projection
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=True)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.dropout_mlp = nn.Dropout(config.dropout)

        # MLP projection
        self.mlp = nn.Linear(config.d_model, config.d_model)

        # BatchNorm (paper: BatchNorm, NOT LayerNorm)
        self.norm_post1 = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(config.d_model, momentum=config.momentum),
            Transpose(1, 2)
        )
        self.norm_post2 = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(config.d_model, momentum=config.momentum),
            Transpose(1, 2)
        )
        self.norm_attn = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(config.d_model, momentum=config.momentum),
            Transpose(1, 2)
        )

        # Feed-forward networks
        self.ff_1 = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=True)
        )
        self.ff_2 = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=True)
        )

        # Dynamic projection (Section 3.3)
        self.dp_rank = config.dp_rank
        self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
        self.dp_v = nn.Linear(self.head_dim, self.dp_rank)

        # Fixed EMA matrix (no learnable parameters — Section 3.2)
        ema_size = max(config.enc_in, config.total_token_number, config.dp_rank)
        ema_matrix = torch.zeros((ema_size, ema_size))
        alpha = config.alpha
        ema_matrix[0][0] = 1
        for i in range(1, ema_size):
            for j in range(i):
                ema_matrix[i][j] = ema_matrix[i-1][j] * (1 - alpha)
            ema_matrix[i][i] = alpha

        self.register_buffer('ema_matrix', ema_matrix)

    def ema(self, src):
        """EMA smoothing (Equation 3). src: (B, nvars, n_heads, T, head_dim)"""
        return torch.einsum('bnhad,ga->bnhgd', src, self.ema_matrix[:src.shape[-2], :src.shape[-2]])

    def dynamic_projection(self, src, mlp):
        """Dynamic projection (Equations 6-7). Reduces channels to dp_rank."""
        src_dp = mlp(src)
        src_dp = F.softmax(src_dp, dim=-1)
        src_dp = torch.einsum('bnhef,bnhec->bnhcf', src, src_dp)
        return src_dp

    def forward(self, src):
        """
        Args:
            src: (batch, nvars, num_tokens, d_model)
        Returns:
            output: same shape as src
        """
        B, nvars, H, C = src.shape

        # Q, K, V projections: (batch, nvars, n_heads, num_tokens, head_dim)
        qkv = self.qkv(src).reshape(B, nvars, H, 3, self.n_heads, C // self.n_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if not self.over_channel:
            # -------------------------------------------------------
            # TOKEN ATTENTION (Section 3.2, Equations 3-5)
            # -------------------------------------------------------
            # A = softmax( EMA(Q) @ EMA(K)^T / sqrt(head_dim) )
            attn_score_along_token = torch.einsum(
                'bnhed,bnhfd->bnhef',
                self.ema(q), self.ema(k)
            ) / (self.head_dim ** 0.5)  # divide by sqrt(d) ✅

            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1))
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v)

        else:
            # -------------------------------------------------------
            # CHANNEL ATTENTION (Section 3.3, Equations 3, 6-7)
            # -------------------------------------------------------
            # Dynamic projection for K and V
            v_dp = self.dynamic_projection(v, self.dp_v)
            k_dp = self.dynamic_projection(k, self.dp_k)

            # EMA-smoothed attention
            attn_score_along_token = torch.einsum(
                'bnhed,bnhfd->bnhef',
                self.ema(q), self.ema(k_dp)
            ) / (self.head_dim ** 0.5)  # divide by sqrt(d) ✅ (was -0.5, now fixed)

            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1))
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v_dp)

        # -------------------------------------------------------
        # HIDDEN-DIMENSION ATTENTION (Section 3.2, Equation 4)
        # -------------------------------------------------------
        # A = softmax( Q^T @ K / sqrt(N) )
        attn_score_along_hidden = torch.einsum(
            'bnhae,bnhaf->bnhef',
            q, k
        ) / (q.shape[-2] ** 0.5)  # divide by sqrt(N) ✅

        attn_along_hidden = self.attn_dropout(F.softmax(attn_score_along_hidden, dim=-1))
        output_along_hidden = torch.einsum('bnhef,bnhaf->bnhae', attn_along_hidden, v)

        # -------------------------------------------------------
        # TOKEN BLEND (Section 3.4) — HEAD LEVEL
        # -------------------------------------------------------
        output1 = rearrange(
            output_along_token.reshape(B * nvars, -1, self.head_dim),
            'bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d',
            hl1=self.n_heads // self.merge_size,
            hl2=output_along_token.shape[-2],
            hl3=self.merge_size
        ).reshape(B * nvars, -1, self.head_dim * self.n_heads)

        output2 = rearrange(
            output_along_hidden.reshape(B * nvars, -1, self.head_dim),
            'bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d',
            hl1=self.n_heads // self.merge_size,
            hl2=output_along_token.shape[-2],
            hl3=self.merge_size
        ).reshape(B * nvars, -1, self.head_dim * self.n_heads)

        # Post-norm
        output1 = self.norm_post1(output1).reshape(B, nvars, -1, self.n_heads * self.head_dim)
        output2 = self.norm_post2(output2).reshape(B, nvars, -1, self.n_heads * self.head_dim)

        # Add & Norm
        src2 = self.ff_1(output1) + self.ff_2(output2)
        src = src + src2
        src = src.reshape(B * nvars, -1, self.n_heads * self.head_dim)
        src = self.norm_attn(src)
        src = src.reshape(B, nvars, -1, self.n_heads * self.head_dim)

        return src


# ============================================================================
# 4. CARD Main Model
# ============================================================================

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
        self.close_channel_idx = getattr(config, 'close_channel_idx', 3)  # default: channel 3

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
        self.Attentions_over_token = nn.ModuleList([
            CARDAttention(config, over_channel=False)
            for _ in range(config.e_layers)
        ])
        self.Attentions_over_channel = nn.ModuleList([
            CARDAttention(config, over_channel=True)
            for _ in range(config.e_layers)
        ])
        self.Attentions_mlp = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model)
            for _ in range(config.e_layers)
        ])
        self.Attentions_dropout = nn.ModuleList([
            nn.Dropout(config.dropout)
            for _ in range(config.e_layers)
        ])
        self.Attentions_norm = nn.ModuleList([
            nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(config.d_model, momentum=config.momentum),
                Transpose(1, 2)
            )
            for _ in range(config.e_layers)
        ])

        # ------------------------------------------------------------
        # MLP DECODER: (patch_num+1)*d_model → pred_len
        # ------------------------------------------------------------
        self.W_out = nn.Linear(
            (patch_num + 1) * config.d_model,
            config.pred_len
        )

    def forward(self, z):
        """
        Args:
            z: (batch, channels, seq_len)   e.g. (B, 18, 60)

        Returns:
            pred_close       : (batch, pred_len) — denormalized close prices in ₹
            revin_close_stats: (mean, std) each (B, 1) — RevIN stats for close channel.
                               Use these to compute loss on normalized scale (O(1)).
        """
        b, c, s = z.shape

        # ----------------------------------------------------------
        # 1. RevIN NORMALIZE (per-channel over time)
        # ----------------------------------------------------------
        z = self.revin(z, mode='norm')  # (B, C, L) — each channel ~N(0,1)

        # Save close-channel RevIN stats before they can be overwritten
        # self.revin.mean/std: (B, C, 1)  → extract close channel
        revin_close_mean = self.revin.mean[:, self.close_channel_idx, :]  # (B, 1)
        revin_close_std  = self.revin.std[:, self.close_channel_idx, :]   # (B, 1)

        # ----------------------------------------------------------
        # 2. PATCH TOKENIZATION (Equation 1)
        # ----------------------------------------------------------
        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # zcube: (B, C, patch_num, patch_len)

        z_embed = self.input_dropout(self.W_input_projection(zcube)) + self.W_pos_embed
        # z_embed: (B, C, patch_num, d_model)

        # Prepend CLS token
        cls_token = self.cls.expand(b, c, 1, -1)   # (B, C, 1, d_model)
        z_embed = torch.cat((cls_token, z_embed), dim=2)
        # z_embed: (B, C, patch_num+1, d_model)

        # ----------------------------------------------------------
        # 3. DUAL ATTENTION ENCODER
        # ----------------------------------------------------------
        inputs = z_embed
        b2, c2, t, h = inputs.shape

        for a_token, a_channel, mlp, drop, norm in zip(
            self.Attentions_over_token,
            self.Attentions_over_channel,
            self.Attentions_mlp,
            self.Attentions_dropout,
            self.Attentions_norm
        ):
            # Channel attention (over features dimension)
            output_channel = a_channel(inputs.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

            # Token attention (over time tokens)
            output_token = a_token(output_channel)

            # Residual + MLP
            outputs = drop(mlp(output_channel + output_token)) + inputs
            outputs = norm(outputs.reshape(b2 * c2, t, -1)).reshape(b2, c2, t, -1)
            inputs = outputs

        # ----------------------------------------------------------
        # 4. MLP DECODER
        # ----------------------------------------------------------
        # outputs: (B, C, patch_num+1, d_model)
        z_flat = outputs.reshape(b, c, -1)          # (B, C, (patch_num+1)*d_model)
        z_pred_norm = self.W_out(z_flat)            # (B, C, pred_len)

        # Close channel output in NORMALIZED space
        close_norm = z_pred_norm[:, self.close_channel_idx, :]  # (B, pred_len)
        
        # ----------------------------------------------------------
        # CRITICAL FIX: Residual connection from the last known price
        # ----------------------------------------------------------
        # Without this, minimizing MSE on random walks forces the model to 
        # predict 0 in normalized space, which means it predicts the 60-period 
        # moving average (mean-reversion collapse). By adding the last known 
        # price, the model learns the DIFFERENCE from now. A prediction of 0 
        # now correctly maps to the Persistence Model (last known price).
        last_close_norm = z[:, self.close_channel_idx, -1].unsqueeze(-1)  # (B, 1)
        close_norm = last_close_norm + close_norm

        # limit to avoid fp32 overflow
        close_norm = torch.clamp(close_norm, min=-10.0, max=10.0)  # ≈±10σ, safe for all stocks

        # ----------------------------------------------------------
        # 5. RevIN DENORMALIZE — back to original price scale
        # ----------------------------------------------------------
        # Denormalize only the close channel output
        pred_close = close_norm * revin_close_std + revin_close_mean  # (B, pred_len)

        return pred_close, (revin_close_mean, revin_close_std)


# ============================================================================
# 5. Signal Decay Loss (Section 4, Equation 12)
# ============================================================================

class SignalDecayMAE(nn.Module):
    """
    Signal decay Mean Absolute Error.

    Equation 12: min E_A [ 1/L * sum_{l=1}^L l^{-1/2} * |ŷ_{t+l} - y_{t+l}| ]

    Weights near-term predictions higher than far-term.
    NO scale_factor — with proper price targets, loss is naturally in correct range.
    """

    def __init__(self, horizon: int = 15):
        super(SignalDecayMAE, self).__init__()
        self.horizon = horizon

        # Decay weights: l^{-1/2}, normalized to sum=1
        weights = torch.tensor([1.0 / math.sqrt(l + 1) for l in range(horizon)])
        weights = weights / weights.sum()
        self.register_buffer('weights', weights)

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch, horizon) — normalized predicted close prices
            targets:     (batch, horizon) — normalized true close prices
        Returns:
            Scalar loss
        """
        errors = torch.abs(predictions - targets)           # (B, horizon)
        weighted = errors * self.weights.unsqueeze(0)       # (B, horizon)
        return weighted.mean()
