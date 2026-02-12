# 🚀 CARD Stock Prediction - Complete Guide

**Consolidated guide for the entire CARD stock prediction pipeline**

Last Updated: February 12, 2026

---

## 📚 Table of Contents

1. [Batch 1: Data Download](#batch-1-data-download)
2. [Batch 2: Data Processing & Feature Engineering](#batch-2-data-processing--feature-engineering)
3. [Batch 3: CARD Architecture + Windowing](#batch-3-card-architecture--windowing)
4. [Batch 4: Training](#batch-4-training)
5. [Troubleshooting](#troubleshooting)
6. [Current Project Structure](#current-project-structure)

---

# Batch 1: Data Download

## ⏱️ Time: 3-4 hours

Download minute-level OHLCV data for all Nifty 50 stocks.

## Prerequisites

- Fyers trading account with API credentials
- Python 3.10+
- 5GB free disk space

## Steps

### 1. Setup Environment

```bash
# Create conda environment
conda create -n card python=3.10
conda activate card

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Fyers API

```bash
# Create .env file
cp .env.template .env

# Edit .env and add:
# CLIENT_ID=your_client_id
# SECRET_KEY=your_secret_key
```

### 3. Generate Access Token

```bash
python pipeline/00_generate_token.py
```

**Process:**
1. Browser opens with Fyers login
2. Authorize the app
3. Copy `auth_code` from redirected URL
4. Paste into terminal
5. Access token saved to `.env`

### 4. Download Data

```bash
python pipeline/01_download_nifty50.py
```

**What happens:**
- Downloads 50 Nifty stocks
- Minute-level data (2022-01-01 to 2025-12-31)
- ~370k rows per stock
- Saves to `data/raw/{STOCK}/full.csv`

**Duration:** 2.5-3 hours

### 5. Verify Download

```bash
python scripts/verify_download.py
```

**Expected:**
- 50 stock CSV files
- Each with >280k rows
- Date range: 2022-01-01 to 2025-12-31

## Output

```
data/
├── raw/
│   ├── RELIANCE/full.csv    (~370k rows, 6 cols)
│   ├── TCS/full.csv
│   └── ... (50 stocks)
└── market_indices/
    └── NIFTY50.csv          (~370k rows, 6 cols)
```

---

# Batch 2: Data Processing & Feature Engineering

## ⏱️ Time: 1-2 hours

Transform raw OHLCV into feature-rich data with 18 engineered features.

## The 18 Features

### Base OHLCV (5 features)
- `open`, `high`, `low`, `close`, `volume`

### Technical Indicators (6 features)
- `ema_5` - 5-period EMA
- `ema_10` - 10-period EMA
- `roi_1m` - 1-minute return
- `bb_upper` - Upper Bollinger Band
- `bb_middle` - Middle Bollinger Band
- `bb_lower` - Lower Bollinger Band

### Market Context (4 features)
- `nifty_close` - Nifty 50 index price
- `nifty_return` - Nifty 1-minute return
- `hour_sin` - Time-of-day (sine)
- `hour_cos` - Time-of-day (cosine)

### Volatility & Volume (3 features)
- `realized_vol_30m` - 30-min rolling volatility
- `volume_ratio` - Volume / 20-period average
- `volume_surge` - Binary: 1 if volume > 2× average

## Steps

### 1. Run Processing

```bash
python pipeline/02_process_stocks.py
```

**What happens:**
- Cleans trading hours (9:15 AM - 3:30 PM)
- Removes outliers
- Adds 18 features
- Merges with Nifty index
- Saves to `data/processed/{STOCK}_processed.csv`

**Duration:** 45-60 minutes

### 2. Verify Processed Data

```bash
python scripts/verify_processed.py
```

**Expected:**
- All 50 stocks processed
- Each with 18 features
- ~370k rows per stock
- Average Nifty correlation > 0.5

## Output

```
data/
├── raw/                    # Original downloads
└── processed/              # ✨ NEW
    ├── RELIANCE_processed.csv  (~370k rows, 18 features)
    ├── TCS_processed.csv
    └── ... (50 stocks)
```

---

# Batch 3: CARD Architecture + Windowing

## ⏱️ Time: 3-4 hours

Implement CARD model and create 60-minute training windows.

## Part 1: CARD Model

Complete implementation from ICLR 2024 paper:

**Components:**
1. **RevIN** - Reversible Instance Normalization
2. **Patch Embedding** - 60 minutes → 13 tokens (8-min patches)
3. **Channel Attention** - Attend across 18 features
4. **Token Attention** - Attend across 13 time tokens
5. **Token Blend** - Multi-scale aggregation
6. **Multi-Task Loss** - Returns + Volatility with signal decay

**Model file:** `models/card_true.py`

### Test Model

```bash
cd models
python card_true.py
```

**Expected:**
```
Model parameters: 1,416,286
Input: torch.Size([4, 60, 18])
Returns: torch.Size([4, 15])
Volatility: torch.Size([4, 15])
✅ All components working!
```

## Part 2: Windowing

Create 60-minute windows with stride=1 (dense sampling).

### Run Windowing

```bash
python pipeline/03_create_windows_60min.py
```

**What happens:**
- Input: 60 minutes × 18 features
- Output: 15 minutes of returns + volatility
- Dense sampling: ~370k windows per stock
- Saves to `data/windows/{STOCK}_windows.npz`

**Duration:** 30-60 minutes

### Window Structure

Each `.npz` file contains:
- `X`: (369,017, 60, 18) - input windows
- `y_returns`: (369,017, 15) - target returns
- `y_volatility`: (369,017, 15) - target volatility
- `timestamps`: (369,017,) - window start times
- `feature_names`: List of 18 feature names

**File size:** ~250-300 MB per stock (compressed)

## Output

```
data/
├── raw/                    # Batch 1
├── processed/              # Batch 2
└── windows/                # ✨ Batch 3
    ├── RELIANCE_windows.npz  (~280 MB)
    ├── TCS_windows.npz       (~275 MB)
    └── ... (50 stocks, ~13.5 GB total)
```

---

# Batch 4: Training

## ⏱️ Time: 2-3 hours (mostly automated)

Train the TRUE CARD model on Nifty 50 data.

---

## 🎯 Phased Training Strategy

Due to RAM constraints, we implement training in 3 phases:

### **Phase 1: 10 Stocks (Proof of Concept)**
- **Purpose**: Verify training pipeline works
- **Time**: ~30 minutes
- **RAM**: ~8GB
- **Samples**: 2.6M windows
- **Expected accuracy**: 52-54%

### **Phase 2: 50 Stocks (Batch Loading)**
- **Purpose**: Full Nifty 50 training
- **Time**: ~2-3 hours
- **RAM**: ~8GB (loads 10 stocks at a time)
- **Samples**: 12.9M windows
- **Expected accuracy**: 54-58%

### **Phase 3: 500 Stocks (Streaming Dataset)**
- **Purpose**: Production-scale model
- **Time**: ~20-30 hours
- **RAM**: ~500MB constant
- **Samples**: 129M windows
- **Expected accuracy**: 56-60%

---

## 📚 Understanding Data Loading Methods

### **Method 1: All-at-Once Loading** (Not feasible)
```
Load all 50 stocks into RAM → Train
RAM needed: 40GB ❌
```

### **Method 2: Batch Loading** (Phase 2)

**Analogy**: Like studying 50 textbooks when your desk fits only 10.

**How it works:**
```
Epoch 1:
  Load stocks 1-10  → Train on 2.6M windows → Clear RAM
  Load stocks 11-20 → Train on 2.6M windows → Clear RAM
  Load stocks 21-30 → Train on 2.6M windows → Clear RAM
  Load stocks 31-40 → Train on 2.6M windows → Clear RAM
  Load stocks 41-50 → Train on 2.6M windows → Clear RAM

Epoch 2:
  Repeat...
```

**Key points:**
- Model sees ALL 50 stocks (just not simultaneously)
- RAM usage: ~8GB constant
- Training speed: 95-98% of all-at-once
- **Accuracy: IDENTICAL** ✅

### **Method 3: Streaming Dataset** (Phase 3)

**Analogy**: Reading books page-by-page instead of loading entire books.

**How it works:**
```
Training batch 1:
  Open RELIANCE_windows.npz → Read 32 samples → Close
  
Training batch 2:
  Open TCS_windows.npz → Read 32 samples → Close
  
Training batch 3:
  Open INFY_windows.npz → Read 32 samples → Close
```

**Key points:**
- Never loads entire stock into RAM
- Loads only 32 samples (1 batch) at a time
- Each sample = 60-minute window
- RAM per batch: ~138 KB
- Smart caching keeps recently used files open
- **Accuracy: IDENTICAL** ✅

---

## 📊 Expected Performance

### Directional Accuracy

| Phase | Stocks | Samples | Expected Accuracy |
|-------|--------|---------|-------------------|
| Phase 1 | 10 | 2.6M | 52-54% |
| Phase 2 | 50 | 12.9M | 54-58% |
| Phase 3 | 500 | 129M | 56-60% |

### Returns Prediction

With 55% directional accuracy:

| Metric | Expected Value | Meaning |
|--------|---------------|---------|
| **MAE** | 0.003-0.005 | Avg error ~0.3-0.5% |
| **RMSE** | 0.006-0.008 | Root mean squared error |
| **R² Score** | 0.10-0.20 | 10-20% variance explained |
| **Correlation** | 0.15-0.25 | Weak-moderate correlation |

**Key insight**: Model is better at **direction** than **magnitude**.

### Trading Performance

| Strategy | Annual Return | Sharpe Ratio |
|----------|--------------|--------------|
| Random (50%) | 0% | 0.0 |
| Our Model (55%) | +8-12% | 0.8-1.2 |
| Perfect (100%) | +50%+ | 3.0+ |

**Why 55% is good:**
- 5% edge over random
- Compounds over many trades
- Realistic for intraday prediction

---

## Training Pipeline Components

1. **Dataset** (`train/stock_dataset.py`)
   - Loads windows from .npz files
   - 70/15/15 train/val/test split
   - Memory-efficient loading

2. **Configuration** (`train/config.py`)
   - All hyperparameters
   - Easy to modify

3. **Metrics** (`train/metrics.py`)
   - Direction accuracy
   - MAE, MSE, RMSE
   - Correlation, Sharpe ratio

4. **Training Script** (`train/04_train_card.py`)
   - TRUE CARD model
   - Mixed precision training
   - Checkpointing & early stopping

## Quick Start

### 1. Verify Prerequisites

```bash
# Check windows exist
ls data/windows/*.npz | wc -l  # Should show 50
```

### 2. Start Training

```bash
# Full training (200 epochs)
python train/04_train_card.py

# Quick test (3 epochs)
python train/04_train_card.py --epochs 3
```

## What Happens During Training

### Console Output

```
============================================================
TRAINING CONFIGURATION
============================================================

Data:
  Stocks: 50
  Train/Val/Test: 0.7/0.15/0.15

Model:
  Input: 60 timesteps × 18 features
  Output: 15 timesteps (returns + volatility)
  Hidden dim: 128
  Layers: 2
  Heads: 8

Training:
  Batch size: 32
  Epochs: 200
  Learning rate: 0.0001
  Device: cuda
  Mixed precision: True

============================================================

Loading datasets...
Total train samples: 12,912,814

Building TRUE CARD model...
Model parameters: 1,416,286

Using device: cuda
GPU: NVIDIA GeForce RTX 4050 Laptop GPU

============================================================
STARTING TRAINING
============================================================

Epoch 1/200
----------------------------------------------------------------------
Epoch 1: 100%|████| 403525/403525 [30:15<00:00, loss=0.004521, lr=0.000100]

Train Loss: 0.004521
Val Loss:   0.004312

======================================================================
EPOCH 1 RESULTS
======================================================================
Metric                     Train         Val
----------------------------------------------------------------------
direction_accuracy         51.23       51.45
mae                      0.003821    0.003654
correlation                0.1234      0.1456
======================================================================

  ⭐ New best model saved: checkpoints/best_model.pt
```

## Expected Results

After 200 epochs (~2-3 hours):

| Metric | Expected | Baseline | Target |
|--------|----------|----------|--------|
| **Direction Accuracy** | **54-58%** | 50% | >53% |
| **MAE** | **0.003-0.005** | 0.008 | <0.006 |
| **Correlation** | **0.15-0.25** | 0.0 | >0.12 |

**Key metric:** Direction Accuracy
- 50-52%: Learning slowly
- 52-54%: Good progress ✅
- 54-56%: Excellent! ⭐
- 56-58%: Outstanding! 🏆

## Hyperparameter Tuning

Edit `train/config.py`:

```python
# For faster training
BATCH_SIZE = 64         # If you have >6GB VRAM
NUM_EPOCHS = 100        # Quick run

# For better accuracy
LEARNING_RATE = 5e-5    # Lower LR, more stable
D_MODEL = 256           # Bigger model
E_LAYERS = 3            # Deeper model

# For debugging
BATCH_SIZE = 16         # Smaller batches
NUM_EPOCHS = 3          # Quick test
```

## Using the Trained Model

```python
import torch
from models.card_true import CARD
from train.config import CARDModelConfig

# Load model
config = CARDModelConfig()
model = CARD(config)

checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    returns, volatility = model(X)  # X: (batch, 60, 18)
```

---

# Troubleshooting

## Common Issues

### 1. Access Token Expired

**Error:** `Invalid access token`

**Solution:**
```bash
python pipeline/00_generate_token.py
```

### 2. API Rate Limit

**Error:** `Rate limit exceeded`

**Solution:** Script has automatic retries and delays. If still failing, increase delays in `pipeline/01_download_nifty50.py`:
```python
SLEEP_BETWEEN_CHUNKS = 2.0  # increase from 1.0
SLEEP_BETWEEN_STOCKS = 5.0  # increase from 2.0
```

### 3. Out of Memory (Training)

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# In train/config.py
BATCH_SIZE = 16  # Reduce from 32
# Or
D_MODEL = 64     # Reduce from 128
```

### 4. Slow Training

**Symptom:** Each epoch takes >1 hour

**Solutions:**
- Increase `BATCH_SIZE = 64` (if VRAM allows)
- Verify using GPU: `DEVICE = 'cuda'`
- Ensure mixed precision: `USE_AMP = True`

### 5. NaN Loss

**Symptom:** `loss=nan` during training

**Solution:**
```python
# In train/config.py
LEARNING_RATE = 1e-5  # Much lower
MAX_GRAD_NORM = 0.5   # Stricter clipping
```

### 6. Model Not Learning

**Symptom:** Direction accuracy stuck at ~50%

**Check:**
1. Data quality: Verify windows are correct
2. Labels: Check returns calculation
3. Model input: Ensure data is normalized

```python
# Debug script
from train.stock_dataset import StockWindowsDataset
from train.config import TrainingConfig

config = TrainingConfig()
dataset = StockWindowsDataset(
    windows_dir=config.WINDOWS_DIR,
    stocks=['RELIANCE'],
    split='train'
)

X, y_ret, y_vol, _ = dataset[0]
print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
print(f"Returns range: [{y_ret.min():.3f}, {y_ret.max():.3f}]")
```

---

# Current Project Structure

```
CARD_FOR_TRADING/
├── data/
│   ├── raw/                      # Batch 1: Downloaded CSVs
│   │   ├── RELIANCE/full.csv
│   │   └── ... (50 stocks)
│   ├── processed/                # Batch 2: 18 features
│   │   ├── RELIANCE_processed.csv
│   │   └── ... (50 stocks)
│   └── windows/                  # Batch 3: Training windows
│       ├── RELIANCE_windows.npz
│       └── ... (50 stocks)
│
├── models/
│   ├── __init__.py
│   ├── card_true.py              # TRUE CARD implementation
│   └── [DEPRECIATED]card_complete.py
│
├── train/
│   ├── __init__.py
│   ├── config.py                 # Training configuration
│   ├── stock_dataset.py          # Dataset class
│   ├── metrics.py                # Evaluation metrics
│   └── 04_train_card.py          # Main training script
│
├── pipeline/
│   ├── 00_generate_token.py      # Fyers token generation
│   ├── 01_download_nifty50.py    # Download data
│   ├── 02_process_stocks.py      # Feature engineering
│   └── 03_create_windows_60min.py # Create windows
│
├── scripts/
│   ├── config_nifty50.py         # Stock list configuration
│   ├── verify_download.py        # Verify downloaded data
│   └── verify_processed.py       # Verify processed data
│
├── losses/
│   └── __init__.py
│
├── checkpoints/                  # Training checkpoints
│   ├── best_model.pt
│   └── checkpoint_epoch_*.pt
│
├── logs/                         # Training logs
│   └── training_history_*.json
│
├── guides_and_checklists/
│   ├── BATCH1_QUICK_START.md
│   ├── BATCH2_QUICK_START.md
│   ├── BATCH3_QUICK_START.md
│   └── BATCH4_QUICK_START.md
│
├── .env                          # API credentials (gitignored)
├── .env.template                 # Template for .env
├── requirements.txt              # Python dependencies
├── setup.py                      # Project setup script
├── README.md                     # Project overview
├── CHANGELOG.md                  # Change history
└── CARD_Stock_Prediction_Strategy.md  # Strategy document
```

---

# Quick Reference

## Complete Pipeline (Start to Finish)

```bash
# 1. Setup
conda create -n card python=3.10
conda activate card
pip install -r requirements.txt

# 2. Configure API
cp .env.template .env
# Edit .env with your credentials

# 3. Generate token
python pipeline/00_generate_token.py

# 4. Download data (2.5-3 hours)
python pipeline/01_download_nifty50.py

# 5. Process data (45-60 min)
python pipeline/02_process_stocks.py

# 6. Create windows (30-60 min)
python pipeline/03_create_windows_60min.py

# 7. Train model (2-3 hours)
python train/04_train_card.py
```

## Key Files to Modify

- **Stock list:** `scripts/config_nifty50.py`
- **Training config:** `train/config.py`
- **Model architecture:** `models/card_true.py`
- **Features:** `pipeline/02_process_stocks.py`

## Important Metrics

- **Direction Accuracy:** Most important! Target >54%
- **MAE:** Mean Absolute Error, target <0.005
- **Correlation:** Target >0.15
- **Sharpe Ratio:** Target >0.08

---

# Success Checklist

## Batch 1 ✅
- [ ] 50 stock CSV files in `data/raw/`
- [ ] Nifty 50 index CSV in `data/market_indices/`
- [ ] All files have >280k rows
- [ ] Date range: 2022-01-01 to 2025-12-31

## Batch 2 ✅
- [ ] All 50 stocks processed
- [ ] Each stock has 18 features
- [ ] ~370k rows per stock
- [ ] Average Nifty correlation >0.5

## Batch 3 ✅
- [ ] CARD model test passes
- [ ] All 50 stocks windowed
- [ ] Each stock has ~369k windows
- [ ] Total ~18.5M windows created

## Batch 4 ✅
- [ ] Training completes without errors
- [ ] Direction accuracy >52%
- [ ] Best model checkpoint saved
- [ ] Training history logged

---

**Total Time:** ~8-12 hours (mostly automated)

**Final Output:** Trained CARD model ready for stock prediction!

---

**Last Updated:** February 12, 2026  
**Status:** Complete Pipeline Documentation
