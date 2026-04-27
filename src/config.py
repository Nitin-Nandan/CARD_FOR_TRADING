"""
Technical Configuration for CARD Model
Production-grade parameters for Nifty 500 predictive framework.
"""

import torch


class Config:
    """
    Standard Production Configuration.
    Optimized for high-frequency predictive performance.
    """

    # ============================================
    # MODEL ARCHITECTURE (CARD ICLR'24 - Don't Change)
    # ============================================
    enc_in = 81  # Input channels (78 features + 3 regimes)
    seq_len = 60  # Lookback window (60 minutes)
    pred_len = 15  # Prediction horizon (15 minutes)
    patch_len = 8  # Patch size for tokenization
    stride = 4  # Patch stride
    d_model = 128  # Model dimension
    n_heads = 8  # Number of attention heads
    e_layers = 2  # Number of encoder layers
    d_ff = 512  # Feedforward dimension
    merge_size = 2  # Token blend merge size
    dp_rank = 8  # Dynamic projection rank
    alpha = 0.9  # EMA alpha for attention smoothing
    dropout = 0.1  # Increased for PRODUCTION (0.1)
    momentum = 0.1  # BatchNorm momentum
    close_channel_idx = 3  # Index of close price channel

    # Aliases
    ENC_IN = enc_in
    SEQ_LEN = seq_len
    PRED_LEN = pred_len
    PATCH_LEN = patch_len
    STRIDE = stride
    D_MODEL = d_model
    N_HEADS = n_heads
    E_LAYERS = e_layers
    D_FF = d_ff
    MERGE_SIZE = merge_size
    DP_RANK = dp_rank
    ALPHA = alpha
    DROPOUT = dropout
    MOMENTUM = momentum

    # ============================================
    # 6-DAY PRODUCTION HYPERPARAMETERS
    # ============================================

    # Learning Rate (STABLE for long training)
    BASE_LR = 1e-3  # Reduced from 3e-3 for stability
    MIN_LR = 1e-6

    # Loss Components
    DIRECTIONAL_WEIGHT = 10.0
    MAGNITUDE_SCALE = 1000.0

    # Regularization (STRONGER for more data)
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 5.0  # Reduced from 10.0 for stability

    # Batch Size
    batch_size = 128
    BATCH_SIZE = batch_size

    # ============================================
    # OPTIMIZATIONS
    # ============================================

    USE_AMP = True
    ACCUMULATION_STEPS = 2
    WARMUP_EPOCHS = 5  # Longer warmup for 6-day stability
    LR_SCHEDULE = "cosine"

    # Data Volume (REALISTIC for 6 days)
    BALANCE_STOCKS = True
    MAX_WINDOWS_PER_STOCK = 50000

    # ============================================
    # 6-DAY TRAINING LIMITS
    # ============================================

    MAX_EPOCHS = 35  # 35 * 4.1h = ~143h (6 days)
    EARLY_STOP_PATIENCE = 12

    # ============================================
    # CHECKPOINTING
    # ============================================

    SAVE_BEST_LOSS = True
    SAVE_BEST_DIR_ACC = True
    SAVE_PERIODIC = True
    PERIODIC_SAVE_INTERVAL = 5
    SAVE_OPTIMIZER = True

    # ============================================
    # MONITORING
    # ============================================

    LOG_INTERVAL = 500
    VAL_INTERVAL = 1

    # ============================================
    # PATHS
    # ============================================

    CHECKPOINT_DIR = "checkpoints/production_6d"
    LOG_DIR = "logs/production_6d"

    # ============================================
    # DATASET (WINDOWS-SAFE)
    # ============================================

    NUM_WORKERS = 0
    PIN_MEMORY = True

    # ============================================
    # DEVICE
    # ============================================

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================
    # REPRODUCIBILITY
    # ============================================

    SEED = 42
    STOCKS = None

    def __repr__(self):
        """Print config as dict"""
        config_dict = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return "\n".join(f"  {k}: {v}" for k, v in config_dict.items())


# For hyperparameter validation (Part 2)
class QuickTestConfig(Config):
    MAX_EPOCHS = 10
    ACCUMULATION_STEPS = 2
    WARMUP_EPOCHS = 2
    EARLY_STOP_PATIENCE = 999
    SAVE_PERIODIC = False
    CHECKPOINT_DIR = "checkpoints/hyperparam_test"
