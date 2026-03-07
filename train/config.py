"""
Training Configuration for CARD Stock Prediction
Phase 0 Clean Rewrite — close price prediction (not returns)
"""

import sys
from pathlib import Path
import torch

# Allow importing from scripts/
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from scripts.config_nifty50 import get_stock_symbols, symbol_to_name as _s2n


class TrainingConfig:
    """All hyperparameters in one place."""

    # ========================================================================
    # PATHS
    # ========================================================================
    PROJECT_ROOT   = Path(__file__).parent.parent
    DATA_DIR       = PROJECT_ROOT / "data"
    WINDOWS_DIR    = DATA_DIR / "windows"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    LOGS_DIR       = PROJECT_ROOT / "logs"

    # ========================================================================
    # STOCKS
    # All 50 Nifty stocks — ALL loaded every epoch via window sampling.
    # RAM note:
    #   MAX_WINDOWS_PER_STOCK=2500  →  50 × 2500 × 60 × 18 × 4 bytes ≈ 540 MB train
    #   Val uses 500 windows/stock × 50 → ≈ 108 MB
    #   Total peak: ≈ 650 MB  (well within 8 GB available)
    # ========================================================================
    STOCKS = [_s2n(s) for s in get_stock_symbols()]   # all 50 Nifty stocks

    # Windows sampled per stock per epoch (train).
    # 20000/stock × 50 stocks = 1M train windows, ~55 min/epoch
    # RAM: 50 × 20k × 60 × 18 × 4 bytes ≈ 4.38 GB (Raw arrays) + Python overhead ≈ 5 GB
    MAX_WINDOWS_PER_STOCK     = 20000
    # Windows sampled per stock for validation
    MAX_VAL_WINDOWS_PER_STOCK = 1000

    # Data splits — by TIME (not random, to prevent leakage)
    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15

    # ========================================================================
    # MODEL ARCHITECTURE (TRUE CARD — unchanged from paper)
    # ========================================================================
    SEQ_LEN  = 60    # 60-min input context
    PRED_LEN = 15    # 15-min price forecast
    ENC_IN   = 18    # number of input channels (features)

    CLOSE_CHANNEL_IDX = 3   # index of 'close' in feature list (0-indexed)
                              # verify this matches your actual feature order!

    D_MODEL    = 128  # hidden dimension
    N_HEADS    = 8    # attention heads
    E_LAYERS   = 2    # encoder layers
    D_FF       = 512  # FFN width (= D_MODEL * 4)
    PATCH_LEN  = 8    # patch length (minutes per patch)
    STRIDE     = 4    # patch stride

    DROPOUT    = 0.1
    MERGE_SIZE = 2    # token blend size
    DP_RANK    = 8    # dynamic projection rank
    ALPHA      = 0.9  # fixed EMA parameter
    MOMENTUM   = 0.1  # BatchNorm momentum

    # ========================================================================
    # TRAINING
    # ========================================================================
    BATCH_SIZE     = 64
    NUM_EPOCHS     = 200   # 5 stocks/epoch × 150 epochs → ~15 passes per stock
    LEARNING_RATE  = 1e-4
    WEIGHT_DECAY   = 1e-5

    # LR schedule
    LR_SCHEDULER  = 'cosine'
    WARMUP_EPOCHS = 3
    MIN_LR        = 1e-6

    # Gradient clipping
    MAX_GRAD_NORM = 1.0

    # Mixed precision
    USE_AMP = False   # fp16 causes overflow in CARD attention; RTX 4050 TF32 is fast enough

    # ========================================================================
    # CHECKPOINTING & LOGGING
    # ========================================================================
    SAVE_FREQ   = 5
    KEEP_LAST_N = 3
    SAVE_BEST   = True
    PRINT_FREQ  = 50

    # Early stopping and Loss
    EARLY_STOP_PATIENCE = 30    # ~5hrs headroom at 10min/epoch
    EARLY_STOP_DELTA    = 0     # accept ANY val improvement (model improves very slowly)
    DIR_LOSS_WEIGHT     = 0.5   # Weight for Directional Penalty Loss

    # ========================================================================
    # HARDWARE
    # ========================================================================
    if torch.cuda.is_available():
        DEVICE   = 'cuda'
        GPU_NAME = torch.cuda.get_device_name(0)
        print(f"✅ GPU: {GPU_NAME}")
    else:
        DEVICE = 'cpu'
        print("⚠️  No GPU detected — training will be slow.")

    NUM_WORKERS = 0      # Windows: keep 0
    PIN_MEMORY  = (DEVICE == 'cuda')

    SEED          = 42
    DETERMINISTIC = False

    # ========================================================================
    # EVALUATION METRICS
    # ========================================================================
    METRICS = [
        'direction_accuracy',
        'mae',
        'correlation',
    ]

    @classmethod
    def print_config(cls):
        print("="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        print(f"Stocks    : {len(cls.STOCKS)} | "
              f"{cls.MAX_WINDOWS_PER_STOCK} windows/stock = "
              f"{len(cls.STOCKS)*cls.MAX_WINDOWS_PER_STOCK:,} total/epoch")
        print(f"Input     : {cls.SEQ_LEN} min × {cls.ENC_IN} features")
        print(f"Output    : {cls.PRED_LEN} close prices (channel {cls.CLOSE_CHANNEL_IDX})")
        print(f"D_MODEL   : {cls.D_MODEL}, Heads: {cls.N_HEADS}, Layers: {cls.E_LAYERS}")
        print(f"Batch     : {cls.BATCH_SIZE}, Epochs: {cls.NUM_EPOCHS}")
        print(f"LR        : {cls.LEARNING_RATE}")
        print(f"Device    : {cls.DEVICE}")
        print("="*60)

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in cls.__dict__.items()
                if k.isupper() and not k.startswith('_')}


# ============================================================================
# CARD Model Config
# ============================================================================

class CARDModelConfig:
    """Thin adapter from TrainingConfig → CARD model __init__ kwargs."""

    def __init__(self, training_config=None):
        cfg = training_config or TrainingConfig

        self.seq_len            = cfg.SEQ_LEN
        self.pred_len           = cfg.PRED_LEN
        self.enc_in             = cfg.ENC_IN
        self.close_channel_idx  = cfg.CLOSE_CHANNEL_IDX
        self.d_model            = cfg.D_MODEL
        self.n_heads            = cfg.N_HEADS
        self.e_layers           = cfg.E_LAYERS
        self.d_ff               = cfg.D_FF
        self.patch_len          = cfg.PATCH_LEN
        self.stride             = cfg.STRIDE
        self.dropout            = cfg.DROPOUT
        self.merge_size         = cfg.MERGE_SIZE
        self.dp_rank            = cfg.DP_RANK
        self.alpha              = cfg.ALPHA
        self.momentum           = cfg.MOMENTUM
        self.task_name          = 'forecast'

        self.total_token_number = int(
            (self.seq_len - self.patch_len) / self.stride + 1
        ) + 1  # +1 for CLS token


if __name__ == "__main__":
    TrainingConfig.print_config()
    mc = CARDModelConfig()
    print(f"Total tokens : {mc.total_token_number}")
    print(f"Close channel: {mc.close_channel_idx}")