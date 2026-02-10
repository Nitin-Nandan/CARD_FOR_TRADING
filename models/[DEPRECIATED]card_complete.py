"""
CARD Stock Prediction - Complete Model Architecture

This file contains all CARD components for easy deployment.
After downloading, split into separate files:
  - models/revin.py
  - models/channel_attention.py
  - models/token_attention.py
  - models/token_blend.py
  - models/card_stock.py
  - losses/multi_task_loss.py

Or use as-is for quick testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# 1. RevIN (Reversible Instance Normalization)
# ============================================================================

class RevIN(nn.Module):
    """Reversible Instance Normalization"""
    
    def __init__(self, num_features, eps=1e-5, affine=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"mode must be 'norm' or 'denorm', got {mode}")
    
    def _normalize(self, x):
        # x: (batch, seq_len, num_features)
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        
        x_norm = (x - self.mean) / self.std
        
        if self.affine:
            x_norm = x_norm * self.affine_weight.view(1, 1, -1) + self.affine_bias.view(1, 1, -1)
        
        return x_norm
    
    def _denormalize(self, x):
        # x: (batch, pred_len) or (batch, pred_len, num_features)
        if x.dim() == 2:
            mean = self.mean[:, :, 0]
            std = self.std[:, :, 0]
        else:
            mean = self.mean
            std = self.std
        
        if self.affine:
            if x.dim() == 2:
                x = (x - self.affine_bias[0]) / self.affine_weight[0]
            else:
                x = (x - self.affine_bias.view(1, 1, -1)) / self.affine_weight.view(1, 1, -1)
        
        return x * std + mean


# ============================================================================
# 2. Channel Attention (Attention across features)
# ============================================================================

class ChannelAttention(nn.Module):
    """Multi-head attention across channels (features)"""
    
    def __init__(self, num_features, d_model, n_heads=8, dropout=0.1):
        super(ChannelAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.num_features = num_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (batch, num_features, num_tokens, d_model)
        batch, num_features, num_tokens, d_model = x.shape
        
        residual = x
        
        # Pool over tokens to get feature representations
        x_pooled = x.mean(dim=2)  # (batch, num_features, d_model)
        
        # QKV
        Q = self.q_proj(x_pooled)
        K = self.k_proj(x_pooled)
        V = self.v_proj(x_pooled)
        
        # Multi-head reshape
        Q = Q.view(batch, num_features, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, num_features, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, num_features, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, num_features, d_model)
        
        # Project
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)
        
        # Broadcast back to tokens
        attn_output = attn_output.unsqueeze(2).expand(-1, -1, num_tokens, -1)
        
        # Residual + norm
        output = self.norm(residual + attn_output)
        
        return output


class ChannelAttentionBlock(nn.Module):
    """Channel attention with FFN"""
    
    def __init__(self, num_features, d_model, n_heads=8, d_ff=512, dropout=0.1):
        super(ChannelAttentionBlock, self).__init__()
        
        self.channel_attn = ChannelAttention(num_features, d_model, n_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = self.channel_attn(x)
        residual = x
        x = self.ffn(x)
        x = self.norm(residual + x)
        return x


# ============================================================================
# 3. Token Attention (Attention across time)
# ============================================================================

class TokenAttention(nn.Module):
    """Multi-head attention across time tokens"""
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(TokenAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (batch, num_features, num_tokens, d_model)
        batch, num_features, num_tokens, d_model = x.shape
        
        residual = x
        
        # Flatten features and tokens together
        x_flat = x.reshape(batch, num_features * num_tokens, d_model)
        
        # QKV
        Q = self.q_proj(x_flat)
        K = self.k_proj(x_flat)
        V = self.v_proj(x_flat)
        
        # Multi-head
        Q = Q.view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, -1, d_model)
        
        # Project
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)
        
        # Reshape back
        attn_output = attn_output.view(batch, num_features, num_tokens, d_model)
        
        # Residual + norm
        output = self.norm(residual + attn_output)
        
        return output


class TokenAttentionBlock(nn.Module):
    """Token attention with FFN"""
    
    def __init__(self, d_model, n_heads=8, d_ff=512, dropout=0.1):
        super(TokenAttentionBlock, self).__init__()
        
        self.token_attn = TokenAttention(d_model, n_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = self.token_attn(x)
        residual = x
        x = self.ffn(x)
        x = self.norm(residual + x)
        return x


# ============================================================================
# 4. Token Blend (Multi-scale aggregation)
# ============================================================================

class TokenBlend(nn.Module):
    """Token blending for multi-scale representations"""
    
    def __init__(self, d_model, blend_size=2, dropout=0.1):
        super(TokenBlend, self).__init__()
        
        self.d_model = d_model
        self.blend_size = blend_size
        
        # Learnable blend weights
        self.blend_weights = nn.Parameter(torch.ones(blend_size) / blend_size)
        
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (batch, num_features, num_tokens, d_model)
        batch, num_features, num_tokens, d_model = x.shape
        
        new_num_tokens = num_tokens // self.blend_size
        
        if new_num_tokens == 0:
            return x
        
        # Group adjacent tokens
        x_grouped = x[:, :, :new_num_tokens * self.blend_size, :].reshape(
            batch, num_features, new_num_tokens, self.blend_size, d_model
        )
        
        # Apply weights
        weights = F.softmax(self.blend_weights, dim=0).view(1, 1, 1, self.blend_size, 1)
        x_blended = (x_grouped * weights).sum(dim=3)
        
        # Project
        x_blended = self.proj(x_blended)
        x_blended = self.dropout(x_blended)
        x_blended = self.norm(x_blended)
        
        return x_blended


# ============================================================================
# 5. Signal Decay Loss (Multi-task with returns + volatility)
# ============================================================================

class SignalDecayMAE(nn.Module):
    """MAE with signal decay weighting"""
    
    def __init__(self, horizon=15):
        super(SignalDecayMAE, self).__init__()
        self.horizon = horizon
        
        weights = torch.tensor([1.0 / math.sqrt(t + 1) for t in range(horizon)])
        weights = weights / weights.sum()
        self.register_buffer('weights', weights)
    
    def forward(self, predictions, targets):
        errors = torch.abs(predictions - targets)
        weighted_errors = errors * self.weights.unsqueeze(0)
        return weighted_errors.mean()


class SignalDecayMSE(nn.Module):
    """MSE with signal decay weighting"""
    
    def __init__(self, horizon=15):
        super(SignalDecayMSE, self).__init__()
        self.horizon = horizon
        
        weights = torch.tensor([1.0 / math.sqrt(t + 1) for t in range(horizon)])
        weights = weights / weights.sum()
        self.register_buffer('weights', weights)
    
    def forward(self, predictions, targets):
        errors = (predictions - targets) ** 2
        weighted_errors = errors * self.weights.unsqueeze(0)
        return weighted_errors.mean()


class MultiTaskLoss(nn.Module):
    """Combined loss for returns and volatility"""
    
    def __init__(self, horizon=15, alpha=0.7, beta=0.3):
        super(MultiTaskLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        
        self.returns_loss = SignalDecayMAE(horizon)
        self.volatility_loss = SignalDecayMSE(horizon)
    
    def forward(self, pred_returns, pred_volatility, true_returns, true_volatility):
        loss_ret = self.returns_loss(pred_returns, true_returns)
        loss_vol = self.volatility_loss(pred_volatility, true_volatility)
        
        total_loss = self.alpha * loss_ret + self.beta * loss_vol
        
        loss_dict = {
            'total': total_loss.item(),
            'returns': loss_ret.item(),
            'volatility': loss_vol.item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# 6. CARD Stock Model (Main Architecture)
# ============================================================================

class PatchEmbedding(nn.Module):
    """Convert time series into patches"""
    
    def __init__(self, patch_len=8, stride=4, num_features=18, d_model=128):
        super(PatchEmbedding, self).__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        self.num_features = num_features
        self.d_model = d_model
        
        self.proj = nn.Linear(patch_len, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, num_features)
        batch, seq_len, num_features = x.shape
        
        # Calculate number of patches
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        
        # Create patches
        patches = []
        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :]
            patches.append(patch)
        
        # Stack and rearrange
        patches = torch.stack(patches, dim=1)  # (batch, num_patches, patch_len, num_features)
        patches = patches.permute(0, 3, 1, 2)  # (batch, num_features, num_patches, patch_len)
        
        # Project
        embedded = self.proj(patches)  # (batch, num_features, num_patches, d_model)
        
        return embedded


class PositionalEncoding(nn.Module):
    """Learnable positional embeddings"""
    
    def __init__(self, num_features, max_tokens, d_model):
        super(PositionalEncoding, self).__init__()
        
        self.pos_emb = nn.Parameter(torch.randn(1, num_features, max_tokens, d_model) * 0.02)
    
    def forward(self, x):
        batch, num_features, num_tokens, d_model = x.shape
        return x + self.pos_emb[:, :, :num_tokens, :]


class CARDStock(nn.Module):
    """
    Complete CARD model for stock prediction
    
    Architecture:
        Input (60 min, 18 features)
        → RevIN
        → Patch Embedding (8-min patches → 13 tokens)
        → Positional Encoding
        → Channel Attention (2 layers)
        → Token Attention (2 layers)
        → Token Blend
        → Flatten
        → Multi-Task Head (Returns + Volatility)
    """
    
    def __init__(
        self,
        num_features=18,
        seq_len=60,
        pred_len=15,
        d_model=128,
        n_heads=8,
        patch_len=8,
        stride=4,
        num_encoder_layers=2,
        dropout=0.1
    ):
        super(CARDStock, self).__init__()
        
        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Calculate number of tokens
        self.num_tokens = (seq_len - patch_len) // stride + 1
        
        # Components
        self.revin = RevIN(num_features)
        
        self.patch_embed = PatchEmbedding(patch_len, stride, num_features, d_model)
        
        self.pos_encoding = PositionalEncoding(num_features, self.num_tokens, d_model)
        
        self.channel_attention = nn.ModuleList([
            ChannelAttentionBlock(num_features, d_model, n_heads, d_model*4, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.token_attention = nn.ModuleList([
            TokenAttentionBlock(d_model, n_heads, d_model*4, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.token_blend = TokenBlend(d_model, blend_size=2, dropout=dropout)
        
        # After blend: num_tokens becomes num_tokens//2
        final_tokens = self.num_tokens // 2
        flatten_dim = num_features * final_tokens * d_model
        
        # Prediction heads
        self.head_returns = nn.Sequential(
            nn.Linear(flatten_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len)
        )
        
        self.head_volatility = nn.Sequential(
            nn.Linear(flatten_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len),
            nn.Softplus()  # Ensure positive
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, 60, 18)
        
        # RevIN normalize
        x = self.revin(x, mode='norm')
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, 18, 13, 128)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Channel attention
        for layer in self.channel_attention:
            x = layer(x)
        
        # Token attention
        for layer in self.token_attention:
            x = layer(x)
        
        # Token blend
        x = self.token_blend(x)  # (batch, 18, 6, 128)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Predictions
        pred_returns = self.head_returns(x)  # (batch, 15)
        pred_volatility = self.head_volatility(x)  # (batch, 15)
        
        # RevIN denormalize (returns only)
        pred_returns = self.revin(pred_returns, mode='denorm')
        
        return pred_returns, pred_volatility
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Testing CARD Stock Model")
    print("="*60)
    
    # Create model
    model = CARDStock(
        num_features=18,
        seq_len=60,
        pred_len=15,
        d_model=128,
        n_heads=8,
        num_encoder_layers=2
    )
    
    print(f"\nModel parameters: {model.get_num_params():,}")
    
    # Test forward pass
    batch = 4
    x = torch.randn(batch, 60, 18)
    
    with torch.no_grad():
        pred_returns, pred_volatility = model(x)
    
    print(f"\nInput: {x.shape}")
    print(f"Returns: {pred_returns.shape}")
    print(f"Volatility: {pred_volatility.shape}")
    
    # Test loss
    criterion = MultiTaskLoss(horizon=15)
    true_returns = torch.randn(batch, 15) * 0.01
    true_volatility = torch.rand(batch, 15) * 0.02
    
    loss, loss_dict = criterion(pred_returns, pred_volatility, true_returns, true_volatility)
    
    print(f"\nLoss dict: {loss_dict}")
    
    print("\n" + "="*60)
    print("✅ All components working!")
    print("="*60)
