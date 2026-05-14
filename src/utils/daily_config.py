"""
Daily CARD configuration.

Optimized for daily NSE stock prediction:
    enc_in  = 15    (15 stationary features)
    seq_len = 120   (6 months of trading days)
    pred_len = 5    (1 trading week ahead)

All hyperparameters follow the CARD ICLR'24 defaults where applicable,
scaled down for daily data (less data than intraday → smaller model,
higher dropout, lower batch size).
"""

import torch


class DailyConfig:
    # ============================================================
    # MODEL ARCHITECTURE  (CARD ICLR'24 — align with paper)
    # ============================================================
    enc_in = 15       # 15 daily features
    seq_len = 120     # 6 months lookback (≈ 2 earnings cycles)
    pred_len = 5      # 1 trading week ahead (recommended)
    patch_len = 8     # patch size for tokenization (paper default)
    stride = 4        # patch stride (patch_num = (120-8)/4 + 1 = 29)
    d_model = 64      # model dimension (smaller than intraday)
    n_heads = 4       # attention heads (d_model / n_heads = 16 head_dim)
    e_layers = 2      # encoder layers
    d_ff = 256        # feed-forward dim (4× d_model)
    merge_size = 2    # token blend merge factor
    dp_rank = 8       # dynamic projection rank
    alpha = 0.9       # EMA smoothing alpha (paper default)
    dropout = 0.2     # higher than intraday — less data
    momentum = 0.1    # BatchNorm momentum

    # Channel containing the prediction target (log_return_close at idx 0)
    close_channel_idx = 0

    # ============================================================
    # TRAINING
    # ============================================================
    BASE_LR = 3e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 5.0
    BATCH_SIZE = 64
    MAX_EPOCHS = 50
    WARMUP_EPOCHS = 5
    EARLY_STOP_PATIENCE = 10
    ACCUMULATION_STEPS = 1
    USE_AMP = True

    # Loss
    DIRECTIONAL_WEIGHT = 10.0
    MAGNITUDE_SCALE = 1000.0

    # ============================================================
    # DATASET
    # ============================================================
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    # test = remaining 0.15

    # ============================================================
    # CHECKPOINTING & LOGGING
    # ============================================================
    CHECKPOINT_DIR = "checkpoints/daily_v1"
    LOG_DIR = "logs/daily_v1"
    SAVE_BEST_LOSS = True
    SAVE_BEST_DIR_ACC = True
    SAVE_PERIODIC = False
    PERIODIC_SAVE_INTERVAL = 10
    VAL_INTERVAL = 1
    LOG_INTERVAL = 50

    # ============================================================
    # RUNTIME
    # ============================================================
    NUM_WORKERS = 2
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
