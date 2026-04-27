import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .layers import Transpose


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

        # BatchNorm (paper: BatchNorm, NOT LayerNorm)
        self.norm_post1 = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(config.d_model, momentum=config.momentum),
            Transpose(1, 2),
        )
        self.norm_post2 = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(config.d_model, momentum=config.momentum),
            Transpose(1, 2),
        )
        self.norm_attn = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(config.d_model, momentum=config.momentum),
            Transpose(1, 2),
        )

        # Feed-forward networks
        self.ff_1 = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=True),
        )
        self.ff_2 = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=True),
        )

        # Dynamic projection (Section 3.3) - ONLY needed for channel attention
        if self.over_channel:
            self.dp_rank = config.dp_rank
            self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
            self.dp_v = nn.Linear(self.head_dim, self.dp_rank)
        else:
            self.dp_rank = config.dp_rank

        # Fixed EMA matrix (no learnable parameters — Section 3.2)
        ema_size = max(config.enc_in, config.total_token_number, config.dp_rank)
        ema_matrix = torch.zeros((ema_size, ema_size))
        alpha = config.alpha
        ema_matrix[0][0] = 1
        for i in range(1, ema_size):
            for j in range(i):
                ema_matrix[i][j] = ema_matrix[i - 1][j] * (1 - alpha)
            ema_matrix[i][i] = alpha

        self.register_buffer("ema_matrix", ema_matrix)

    def ema(self, src):
        """EMA smoothing (Equation 3). src: (B, nvars, n_heads, T, head_dim)"""
        return torch.einsum(
            "bnhad,ga->bnhgd", src, self.ema_matrix[: src.shape[-2], : src.shape[-2]]
        )

    def dynamic_projection(self, src, mlp):
        """Dynamic projection (Equations 6-7). Reduces channels to dp_rank."""
        src_dp = mlp(src)
        src_dp = F.softmax(src_dp, dim=-1)
        src_dp = torch.einsum("bnhef,bnhec->bnhcf", src, src_dp)
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
        qkv = (
            self.qkv(src)
            .reshape(B, nvars, H, 3, self.n_heads, C // self.n_heads)
            .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if not self.over_channel:
            # TOKEN ATTENTION (Section 3.2, Equations 3-5)
            attn_score_along_token = torch.einsum(
                "bnhed,bnhfd->bnhef", self.ema(q), self.ema(k)
            ) / (self.head_dim**0.5)

            attn_along_token = self.attn_dropout(
                F.softmax(attn_score_along_token, dim=-1)
            )
            output_along_token = torch.einsum("bnhef,bnhfd->bnhed", attn_along_token, v)

        else:
            # CHANNEL ATTENTION (Section 3.3, Equations 3, 6-7)
            v_dp = self.dynamic_projection(v, self.dp_v)
            k_dp = self.dynamic_projection(k, self.dp_k)

            # EMA-smoothed attention
            attn_score_along_token = torch.einsum(
                "bnhed,bnhfd->bnhef", self.ema(q), self.ema(k_dp)
            ) / (self.head_dim**0.5)

            attn_along_token = self.attn_dropout(
                F.softmax(attn_score_along_token, dim=-1)
            )
            output_along_token = torch.einsum(
                "bnhef,bnhfd->bnhed", attn_along_token, v_dp
            )

        # HIDDEN-DIMENSION ATTENTION (Section 3.2, Equation 4)
        attn_score_along_hidden = torch.einsum("bnhae,bnhaf->bnhef", q, k) / (
            q.shape[-2] ** 0.5
        )

        attn_along_hidden = self.attn_dropout(
            F.softmax(attn_score_along_hidden, dim=-1)
        )
        output_along_hidden = torch.einsum("bnhef,bnhaf->bnhae", attn_along_hidden, v)

        # TOKEN BLEND (Section 3.4) — HEAD LEVEL
        output1 = rearrange(
            output_along_token.reshape(B * nvars, -1, self.head_dim),
            "bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d",
            hl1=self.n_heads // self.merge_size,
            hl2=output_along_token.shape[-2],
            hl3=self.merge_size,
        ).reshape(B * nvars, -1, self.head_dim * self.n_heads)

        output2 = rearrange(
            output_along_hidden.reshape(B * nvars, -1, self.head_dim),
            "bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d",
            hl1=self.n_heads // self.merge_size,
            hl2=output_along_token.shape[-2],
            hl3=self.merge_size,
        ).reshape(B * nvars, -1, self.head_dim * self.n_heads)

        # Post-norm
        output1 = self.norm_post1(output1).reshape(
            B, nvars, -1, self.n_heads * self.head_dim
        )
        output2 = self.norm_post2(output2).reshape(
            B, nvars, -1, self.n_heads * self.head_dim
        )

        # Add & Norm
        src2 = self.ff_1(output1) + self.ff_2(output2)
        src = src + src2
        src = src.reshape(B * nvars, -1, self.n_heads * self.head_dim)
        src = self.norm_attn(src)
        src = src.reshape(B, nvars, -1, self.n_heads * self.head_dim)

        return src
