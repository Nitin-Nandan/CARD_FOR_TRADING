"""
Training Configuration for CARD Stock Prediction - FIXED

All hyperparameters and settings in one place.
Modify this file to experiment with different configurations.

FIXES:
- Changed epochs from 50 to 150
- Added early stopping patience of 20
- Fixed GPU detection for RTX 4050
"""

from pathlib import Path
import torch
import sys


class TrainingConfig:
    """Configuration for CARD training"""
    
    # ========================================================================
    # PATHS
    # ========================================================================
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    WINDOWS_DIR = DATA_DIR / "windows"
    MODELS_DIR = PROJECT_ROOT / "models"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # ========================================================================
    # DATA
    # ========================================================================
    # Nifty 50 stocks (all 50)
    # PHASE 1: 10 stocks for proof of concept (diverse sectors)
    # Full 50-stock list commented out - uncomment for Phase 2
    STOCKS = [
        'RELIANCE',    # Energy
        'TCS',         # IT
        'INFY',        # IT
        'HDFCBANK',    # Banking
        'ICICIBANK',   # Banking
        'BHARTIARTL',  # Telecom
        'ITC',         # FMCG
        'SBIN',        # Banking
        'BAJFINANCE',  # Finance
        'MARUTI'       # Auto
    ]
    
    # Full 50 stocks (for Phase 2 - batch loading):
    # STOCKS = [
    #     'ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
    #     'BAJAJFINSV', 'BPCL', 'BHARTIARTL', 'BRITANNIA', 'CIPLA',
    #     'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GRASIM',
    #     'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO',
    #     'HINDUNILVR', 'ICICIBANK', 'ITC', 'INFY',
    #     'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI',
    #     'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE',
    #     'SBILIFE', 'SHRIRAMFIN', 'SBIN', 'SUNPHARMA', 'TCS',
    #     'TATACONSUM', 'TATASTEEL', 'TECHM', 'TITAN',
    #     'TRENT', 'ULTRACEMCO', 'WIPRO', 'ADANIENT',
    #     'BEL', 'INDIGO', 'APOLLOHOSP', 'MAXHEALTH'
    # ]
    
    # Data splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # ========================================================================
    # MODEL ARCHITECTURE (TRUE CARD)
    # ========================================================================
    SEQ_LEN = 60           # Input: 60 minutes
    PRED_LEN = 15          # Output: 15 minutes
    ENC_IN = 18            # Number of features
    
    D_MODEL = 128          # Hidden dimension
    N_HEADS = 8            # Number of attention heads
    E_LAYERS = 2           # Number of encoder layers
    D_FF = 512             # FFN dimension (D_MODEL * 4)
    
    PATCH_LEN = 8          # Patch length (8 minutes)
    STRIDE = 4             # Patch stride
    
    DROPOUT = 0.1          # Dropout rate
    MERGE_SIZE = 2         # Token blend size
    DP_RANK = 8            # Dynamic projection rank
    ALPHA = 0.9            # EMA parameter (fixed)
    MOMENTUM = 0.1         # BatchNorm momentum
    
    # ========================================================================
    # TRAINING - OPTIMIZED FOR MEMORY EFFICIENCY
    # ========================================================================
    BATCH_SIZE = 32        # Reduced from 64 for memory stability
    NUM_EPOCHS = 200       # Increased for thorough training
    LEARNING_RATE = 1e-4   # Initial learning rate
    WEIGHT_DECAY = 1e-5    # L2 regularization
    
    # Learning rate schedule
    WARMUP_EPOCHS = 5      # Linear warmup
    LR_SCHEDULER = 'cosine'  # 'cosine' or 'step'
    MIN_LR = 1e-6          # Minimum learning rate
    
    # Gradient clipping
    MAX_GRAD_NORM = 1.0
    
    # Mixed precision training (faster on RTX 4050)
    USE_AMP = True
    
    # ========================================================================
    # LOSS
    # ========================================================================
    LOSS_ALPHA = 0.7       # Weight for returns loss
    LOSS_BETA = 0.3        # Weight for volatility loss
    
    # ========================================================================
    # CHECKPOINTING
    # ========================================================================
    SAVE_FREQ = 5          # Save checkpoint every N epochs
    KEEP_LAST_N = 3        # Keep last N checkpoints
    SAVE_BEST = True       # Save best validation model
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    LOG_FREQ = 100         # Log every N batches
    VAL_FREQ = 1           # Validate every N epochs
    PRINT_FREQ = 50        # Print progress every N batches
    
    # ========================================================================
    # HARDWARE - FIXED
    # ========================================================================
    # Force CUDA check and provide helpful error message
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        GPU_NAME = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {GPU_NAME}")
    else:
        DEVICE = 'cpu'
        print("⚠️  WARNING: No GPU detected! Training will be VERY slow.")
        print("   Please check:")
        print("   1. CUDA drivers are installed")
        print("   2. PyTorch with CUDA support is installed")
        print("   3. Run: python -c 'import torch; print(torch.cuda.is_available())'")
    
    NUM_WORKERS = 0        # FIXED: Set to 0 for Windows compatibility
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # ========================================================================
    # REPRODUCIBILITY
    # ========================================================================
    SEED = 42
    DETERMINISTIC = False  # Set to True for reproducibility (slower)
    
    # ========================================================================
    # EARLY STOPPING - FIXED
    # ========================================================================
    EARLY_STOP_PATIENCE = 20   # FIXED: Changed from 10 to 20
    EARLY_STOP_DELTA = 1e-4    # Minimum improvement to count
    
    # ========================================================================
    # EVALUATION METRICS
    # ========================================================================
    METRICS = [
        'direction_accuracy',  # % correct direction predictions
        'mae',                 # Mean absolute error
        'mse',                 # Mean squared error
        'correlation',         # Correlation between pred and actual
        'sharpe',             # Sharpe-like metric
    ]
    
    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        print(f"\nData:")
        print(f"  Stocks: {len(cls.STOCKS)}")
        print(f"  Train/Val/Test: {cls.TRAIN_RATIO}/{cls.VAL_RATIO}/{cls.TEST_RATIO}")
        
        print(f"\nModel:")
        print(f"  Input: {cls.SEQ_LEN} timesteps × {cls.ENC_IN} features")
        print(f"  Output: {cls.PRED_LEN} timesteps (returns + volatility)")
        print(f"  Hidden dim: {cls.D_MODEL}")
        print(f"  Layers: {cls.E_LAYERS}")
        print(f"  Heads: {cls.N_HEADS}")
        
        print(f"\nTraining:")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Early stop patience: {cls.EARLY_STOP_PATIENCE}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Device: {cls.DEVICE}")
        if cls.DEVICE == 'cuda':
            print(f"  GPU: {cls.GPU_NAME}")
        print(f"  Mixed precision: {cls.USE_AMP}")
        print(f"  Num workers: {cls.NUM_WORKERS}")
        
        print(f"\nPaths:")
        print(f"  Windows: {cls.WINDOWS_DIR}")
        print(f"  Checkpoints: {cls.CHECKPOINTS_DIR}")
        print(f"  Logs: {cls.LOGS_DIR}")
        print("="*60)
    
    @classmethod
    def to_dict(cls):
        """Convert to dictionary (for saving)"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and key.isupper()
        }


# ============================================================================
# CARD Model Config (for model initialization)
# ============================================================================

class CARDModelConfig:
    """Configuration object for CARD model"""
    
    def __init__(self, training_config=None):
        if training_config is None:
            training_config = TrainingConfig
        
        self.seq_len = training_config.SEQ_LEN
        self.pred_len = training_config.PRED_LEN
        self.enc_in = training_config.ENC_IN
        self.d_model = training_config.D_MODEL
        self.n_heads = training_config.N_HEADS
        self.e_layers = training_config.E_LAYERS
        self.d_ff = training_config.D_FF
        self.patch_len = training_config.PATCH_LEN
        self.stride = training_config.STRIDE
        self.dropout = training_config.DROPOUT
        self.merge_size = training_config.MERGE_SIZE
        self.dp_rank = training_config.DP_RANK
        self.alpha = training_config.ALPHA
        self.momentum = training_config.MOMENTUM
        self.task_name = 'forecast'
        
        # Calculate total tokens
        self.total_token_number = int(
            (self.seq_len - self.patch_len) / self.stride + 1
        ) + 1  # +1 for CLS token


if __name__ == "__main__":
    # Print configuration
    TrainingConfig.print_config()
    
    # Test model config
    model_config = CARDModelConfig()
    print(f"\nModel config created:")
    print(f"  Total tokens: {model_config.total_token_number}")
    print(f"  Task: {model_config.task_name}")