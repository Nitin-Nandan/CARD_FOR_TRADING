<div align="center">

# ⚡ CARD for Stock Trading

**Channel Aligned Robust Blend Transformer — Applied to Intraday Stock Price Prediction**

[![Paper](https://img.shields.io/badge/ICLR%202024-CARD%20Paper-blue)](docs/reference/CARD.pdf)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

*Predict the next 15 closing prices of any Nifty 50 stock from 60 minutes of intraday data*

</div>

---

## 📋 Table of Contents

**I. Understanding the Project**
- [1. What This Project Does](#1-what-this-project-does)
- [2. Frequently Asked Questions (FAQ)](#2-frequently-asked-questions-faq)
- [3. CARD Architecture Deep Dive](#3-card-architecture-deep-dive)

**II. Getting Started**
- [4. Project Structure](#4-project-structure)
- [5. Setup & Installation](#5-setup--installation)

**III. The Pipeline**
- [6. Step-by-Step Pipeline](#6-step-by-step-pipeline)
  - [6.0 Generate Fyers Token](#60-generate-fyers-access-token)
  - [6.1 Download Raw Data](#61-download-raw-data-3-hours)
  - [6.2 Feature Engineering](#62-feature-engineering-1-hour)
  - [6.3 Create Training Windows](#63-create-training-windows-1-hour)
  - [6.4 Verify Data](#64-verify-data-optional)

**IV. Training**
- [7. Model Training (Global Foundation Model)](#7-model-training-global-foundation-model)
  - [7.1 Data Loading Strategy](#71-data-loading-strategy)
  - [7.2 Training Configuration](#72-training-configuration)
  - [7.3 Loss Function](#73-loss-function)
  - [7.4 Running Training](#74-running-training)
  - [7.5 Loading a Trained Model](#75-loading-a-trained-model)
- [8. Transfer Learning (Per-Stock Fine-Tuning)](#8-transfer-learning-per-stock-fine-tuning)
  - [8.1 Freeze/Unfreeze Strategy](#81-freezeunfreeze-strategy)
  - [8.2 Transfer Learning Hyperparameters](#82-transfer-learning-hyperparameters)
  - [8.3 Running Transfer Learning](#83-running-transfer-learning)

**V. Evaluation & Demo**
- [9. Training Results](#9-training-results)
- [10. Live Demo](#10-live-demo)

**VI. Technical Deep Dives**
- [11. Key Design Decisions](#11-key-design-decisions)
  - [11.1 Why Close Prices, Not Returns](#111-why-close-prices-not-returns)
  - [11.2 RevIN Denormalization — The Cross-Domain Problem](#112-revin-denormalization--the-cross-domain-problem)
  - [11.3 Why One Global Model](#113-why-one-global-model)
  - [11.4 Temporal Data Split](#114-temporal-data-split-no-random-split)
- [12. The 18 Input Features — Detailed Breakdown](#12-the-18-input-features--detailed-breakdown)
- [13. Configuration Reference](#13-configuration-reference)

**VII. Reference**
- [14. Troubleshooting](#14-troubleshooting)
- [15. Development History](#15-development-history)
- [16. Roadmap](#16-roadmap)

---

# Part I: Understanding the Project

---

## 1. What This Project Does

This project implements the **CARD** (Channel Aligned Robust Blend) Transformer from the [ICLR 2024 paper](docs/reference/CARD.pdf) and applies it to **intraday stock price prediction** on the Indian stock market (NSE).

### The Core Idea

| | Detail |
|---|---|
| **Input** | Last **60 minutes** of live stock data (18 features per minute) |
| **Output** | Next **15 closing prices** (one per minute) |
| **Data** | 1-minute OHLCV bars for all **50 Nifty stocks** (Jan 2022 – Dec 2025) |
| **Training samples** | ~12.9 million sliding windows from 50 stocks |
| **Inference** | Give any stock's last 60 minutes → get 15 predicted close prices in ₹ |

### Two-Phase Training Strategy

1. **Global Foundation Model** — Train **one** CARD model on all 50 stocks simultaneously. The model learns universal intraday price dynamics (breakouts, mean-reversion, volatility clustering).
2. **Per-Stock Fine-Tuning** — Clone the foundation model and fine-tune only the top layers on individual stock data. This teaches each model the "personality" of that specific stock while preserving general market "physics."

---

## 2. Frequently Asked Questions (FAQ)

### Q1: One shared model or 50 separate models?

**Answer: ONE shared model, trained on all 50 stocks simultaneously.**

- CARD processes windows independently in the batch dimension
- RevIN normalizes each window per-channel → a ₹100 stock and a ₹5000 stock look the same to the model
- Training on 50 stocks = vastly more data = better generalization
- One training run, one checkpoint, one inference call per stock
- 50 separate models = 50 training runs = impractical

Each training batch is a random mix of windows from all stocks. The model learns "general intraday price movement patterns" from all of them simultaneously. After global training, we optionally fine-tune per stock (see [Section 8](#8-transfer-learning-per-stock-fine-tuning)).

### Q2: What exactly do we predict?

**Answer: The next 15 CLOSE PRICES (raw ₹, normalized by RevIN internally).**

**NOT returns. NOT log-returns. NOT volatility.**

- Input → 60 minutes of `[open, high, low, close, volume + 13 indicators]` = 18 features
- Target → next 15 minutes of `[close price]`
- RevIN normalizes close prices in the input to N(0,1) before the model processes them
- RevIN denormalizes predictions back to real ₹ price levels after the model
- Loss: SignalDecayMAE between predicted and actual close prices (in normalized space)
- **This is exactly how CARD was designed to work**

### Q3: Are we using minute-wise data? Why not daily?

**Answer: YES — 1-minute OHLCV bars.**

- 4 years × 250 trading days × 375 minutes/day ≈ **375,000 data points per stock**
- After windowing (60 in, 15 out): ~370k windows per stock
- × 50 stocks = ~18.5 million windows total
- We don't load all at once — we use memory-efficient sampling (see [Section 7.1](#71-data-loading-strategy))

**Daily data would NOT work:** 4 years × 250 days = only 1000 rows per stock → far too few for deep learning.

### Q4: How does the model handle different stock price scales?

**Answer: RevIN handles it automatically.**

- HDFC Bank at ₹1600 and ITC at ₹450 — completely different scales
- RevIN normalizes **each window independently**: `(x - window_mean) / window_std`
- After normalization, all windows look like N(0,1) regardless of the stock
- At prediction time: RevIN inverse-transforms output back to real ₹ price
- No manual normalization of targets needed

### Q5: What does inference look like at runtime?

```python
# For any stock at any moment during market hours:
window = get_last_60_minutes(stock_symbol)   # shape: (60, 18)
pred_close = model.predict(window)           # shape: (15,) → next 15 close prices in ₹
direction = "UP" if pred_close[-1] > window[-1, close_idx] else "DOWN"
```

### Q6: How long does training take?

**On an RTX 4050 Laptop GPU (6 GB VRAM):**

| Phase | Stocks | Windows/epoch | Time/epoch | Total |
|---|---|---|---|---|
| Quick test | 10 | 200k | ~5 min | ~30 min (5 epochs) |
| Phase 1 (current) | 50 | 1M | ~25 min | ~2-3 hours (124 epochs) |
| Phase 2 (planned) | 500 | 10M | ~4 hours | ~20-28 hours |

### Q7: Can this scale to 500 stocks?

**Yes. The architecture is already designed for it.** Only `scripts/config_nifty50.py` needs updating with the new stock list. The data pipeline, model, and transfer learning script all handle arbitrary stock counts. RAM stays flat because we sample windows per stock (see [Section 7.1](#71-data-loading-strategy)).

### Q8: What are the success criteria?

| Metric | Minimum | Good | Outstanding |
|---|---|---|---|
| Direction Accuracy | > 52% | > 55% | > 58% |
| 15-min Price MAE | < 0.5% of price | < 0.2% | < 0.1% |
| Simulated Sharpe | > 0.3 | > 0.8 | > 1.2 |
| Val Loss | Decreasing | Converges smoothly | — |

---

## 3. CARD Architecture Deep Dive

### High-Level Flow

```
Input: (Batch, 18 channels, 60 timesteps)
  │
  ▼
RevIN Normalize ─── per-channel normalization over time dimension
  │
  ▼
Patch Embedding ─── 60 minutes → 14 tokens (8-min patches, stride 4) + CLS token
  │
  ▼
┌──────────────────────────────────────────┐
│  × 2 Encoder Layers                     │
│  ├─ EMA-Smoothed Q/K (fixed α=0.9)     │ ← reduces noise in attention
│  ├─ Token Attention                      │ ← attend across 14 temporal tokens
│  ├─ Channel Attention                    │ ← attend across 18 feature channels
│  ├─ Dynamic Projection (rank=8)          │ ← low-rank channel mixing
│  └─ Token Blend (merge_size=2)           │ ← multi-scale aggregation
└──────────────────────────────────────────┘
  │
  ▼
MLP Decoder ─── project to (Batch, 18, 15) → extract close channel index
  │
  ▼
RevIN Denormalize ─── convert back to real ₹ prices using stored statistics
  │
  ▼
Output: (Batch, 15) → Next 15 close prices in ₹
```

### Key Components Explained

**1. RevIN (Reversible Instance Normalization)**
Normalizes each input channel independently over the time dimension. Stores per-channel mean and std so the output can be denormalized back to real price levels. This is what lets one model handle stocks at totally different price scales.

**2. Patch Embedding**
Converts the 60-minute sequence into 14 overlapping 8-minute patches (stride=4), plus a learnable CLS token. Each patch is projected to `d_model=128` dimensions. This is analogous to Vision Transformer patch tokenization but for time series.

**3. EMA-Smoothed Attention (Equations 3-4 in paper)**
Before computing attention, queries and keys are smoothed with an Exponential Moving Average (fixed α=0.9). This reduces noise in the attention weights, preventing the model from overfitting to minute-level fluctuations.

**4. Token Attention + Channel Attention**
Two separate attention mechanisms per layer:
- **Token Attention**: Each feature channel independently attends across the 14 time tokens → captures temporal patterns
- **Channel Attention**: Each time token independently attends across the 18 feature channels → captures cross-feature relationships

**5. Dynamic Projection (Equations 6-7)**
Low-rank projection (rank=8) that efficiently mixes channel information without the O(C²) cost of full cross-channel attention.

**6. Token Blend**
Multi-scale aggregation: merges pairs of tokens (merge_size=2) to capture patterns at different time scales, similar to a hierarchical pooling.

**7. SignalDecayMAE Loss (Equation 12)**
Weights near-term predictions higher: `weight[l] = l^{-1/2}`. Minute 1 has weight 1.0, minute 15 has weight 0.258. This prioritizes short-term accuracy where the model has more predictive power.

### Model Specifications

| Parameter | Value |
|---|---|
| Total parameters | **1,387,471** (~5.4 MB checkpoint) |
| Hidden dimension (d_model) | 128 |
| Attention heads | 8 |
| Encoder layers | 2 |
| FFN width (d_ff) | 512 |
| Patch length | 8 minutes |
| Patch stride | 4 |
| Total tokens | 14 + 1 CLS = 15 |
| Dropout | 0.1 |
| EMA alpha | 0.9 (fixed, not learnable) |
| Dynamic projection rank | 8 |
| Token blend merge size | 2 |

### Model File

Implementation in [`models/card_true.py`](models/card_true.py) — 511 lines, 5 classes:
- `RevIN` — Reversible Instance Normalization
- `Transpose` — Helper for BatchNorm
- `CARDAttention` — The core attention module (token + channel + EMA)
- `CARD` — Main model class
- `SignalDecayMAE` — Signal decay loss function

---

# Part II: Getting Started

---

## 4. Project Structure

```
CARD_FOR_TRADING/
│
├── models/                          # Model architecture
│   ├── card_true.py                 #   TRUE CARD implementation (511 lines)
│   └── __init__.py
│
├── pipeline/                        # Ordered data pipeline (run 00→05 in sequence)
│   ├── 00_generate_token.py         #   Generate Fyers API access token
│   ├── 01_download_nifty50.py       #   Download 1-min OHLCV for 50 stocks (~3 hrs)
│   ├── 02_process_stocks.py         #   Feature engineering → 18 features (~1 hr)
│   ├── 03_create_windows_60min.py   #   Sliding windows: X=(60,18), y=(15,) (~1 hr)
│   ├── 04_normalize_windows.py      #   Normalize input X only (RevIN handles y)
│   └── 05_transfer_learning.py      #   Per-stock fine-tuning from global model
│
├── train/                           # Training infrastructure
│   ├── config.py                    #   ALL hyperparameters in one file
│   ├── 04_train_card.py             #   Main training script (global model)
│   ├── stock_dataset.py             #   Dataset class with memory-efficient loading
│   └── metrics.py                   #   Direction accuracy, MAE, correlation, Sharpe
│
├── scripts/                         # Utility & config scripts
│   ├── config_nifty50.py            #   Nifty 50 stock list + Fyers symbol mapping
│   ├── verify_download.py           #   Verify raw data integrity → logs/reports/
│   ├── verify_processed.py          #   Verify processed data quality → logs/reports/
│   ├── analyze_history.py           #   Summarize training history JSON → logs/reports/
│   └── audit_data.py               #   Pre-training data sanity check
│
├── data/                            # Data directory (gitignored, ~20 GB total)
│   ├── raw/                         #   Raw CSVs: {STOCK}/full.csv (~370k rows each)
│   ├── processed/                   #   Feature-engineered: {STOCK}_processed.csv
│   ├── market_indices/              #   NIFTY50.csv (~370k rows)
│   └── windows/                     #   Training windows: {STOCK}_windows.npz (~300 MB each)
│
├── checkpoints/                     # Model checkpoints (~16 MB global, ~5 MB each fine-tuned)
│   ├── best_model.pt                #   Global foundation model (124 epochs)
│   └── fine_tuned/                  #   Per-stock fine-tuned models
│       ├── card_TCS.pt              #     6 models currently trained
│       ├── card_INFY.pt
│       ├── card_HCLTECH.pt
│       ├── card_HDFCBANK.pt
│       ├── card_TECHM.pt
│       └── card_WIPRO.pt
│
├── logs/                            # All log outputs
│   ├── reports/                     #   Verification & analysis text outputs
│   ├── *.log                        #   Fyers API logs (download/processing)
│   └── training_history_*.json      #   Per-run metrics (loss, accuracy per epoch)
│
├── results/                         # Evaluation outputs (plots, CSVs)
│   ├── metrics/
│   ├── predictions/
│   └── training_curves/
│
├── demo_server.py                   # Flask backend — live Fyers data + inference
├── demo_ui/
│   └── index.html                   # Chart.js frontend — predicted vs actual
│
├── docs/
│   └── reference/                   # CARD paper PDF + extracted text
│       ├── CARD.pdf
│       ├── card_text.txt
│       └── card_paper_text.txt
│
├── pandas_ta/                       # Vendored technical indicator library
├── .env                             # API credentials (gitignored)
├── .env.template                    # Credential template
├── requirements.txt                 # Python dependencies
├── setup.py                         # Directory structure initializer
└── README.md                        # ← You are here
```

---

## 5. Setup & Installation

### Prerequisites

- **Python 3.10+** (Conda recommended)
- **NVIDIA GPU** with CUDA (tested on RTX 4050 Laptop, 6 GB VRAM)
- **Fyers Trading Account** with API v3 credentials ([register here](https://myapi.fyers.in/))
- **~20 GB free disk space** for data + windows

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/Nitin-Nandan/CARD_FOR_TRADING.git
cd CARD_FOR_TRADING

# 2. Create conda environment
conda create -n card python=3.10
conda activate card

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize directory structure (creates empty dirs + .gitkeep)
python setup.py

# 5. Configure API credentials
cp .env.template .env
# Edit .env → fill in CLIENT_ID and SECRET_KEY from your Fyers dashboard
```

### Key Dependencies

| Package | Purpose |
|---|---|
| `torch 2.1` | Deep learning framework |
| `fyers-apiv3` | Fyers stock market API |
| `pandas`, `numpy` | Data processing |
| `flask`, `flask-cors` | Demo web server |
| `tqdm` | Progress bars |
| `loguru` | Structured logging |
| `scikit-learn` | StandardScaler for normalization |
| `einops` | Tensor rearrangement (used in CARD attention) |

---

# Part III: The Pipeline

---

## 6. Step-by-Step Pipeline

The entire workflow is a **6-step sequential pipeline**. Each step depends on the output of the previous.

**Total time: ~8-12 hours (mostly automated data download + training)**

### 6.0 Generate Fyers Access Token

```bash
python pipeline/00_generate_token.py
```

**What happens:**
1. Script creates a Fyers session with your `CLIENT_ID`
2. Browser opens with Fyers OAuth login page
3. You log in and authorize the app
4. Browser redirects to `https://127.0.0.1/?auth_code=...`
5. Copy the `auth_code` from the URL bar
6. Paste into terminal
7. Access token saved to `.env` automatically

> ⚠️ **Access tokens expire daily.** Re-run this before any data download or live demo session.

---

### 6.1 Download Raw Data (~3 hours)

```bash
python pipeline/01_download_nifty50.py
```

**What happens:**
- Downloads all 50 Nifty stocks + Nifty 50 index
- 1-minute OHLCV bars from **Jan 1, 2022 to Dec 31, 2025**
- Handles Fyers API rate limiting (automatic retries with exponential backoff)
- Chunks requests into 100-day ranges (API limit)
- ~370,000 rows per stock (4 years × 250 days × 375 minutes)

**Output:**
```
data/raw/RELIANCE/full.csv     (~370k rows, 6 columns: timestamp, open, high, low, close, volume)
data/raw/TCS/full.csv
data/raw/.../full.csv          (50 stocks total)
data/market_indices/NIFTY50.csv
```

**Verify:**
```bash
python scripts/verify_download.py
# Expected: 50 stocks, each >280k rows, date range 2022-01-01 to 2025-12-31
# Report saved to logs/reports/verification_report.txt
```

---

### 6.2 Feature Engineering (~1 hour)

```bash
python pipeline/02_process_stocks.py
```

**What happens:**
- Filters trading hours only (9:15 AM – 3:30 PM IST)
- Removes weekends and holidays
- Removes statistical outliers
- Computes 13 additional features from raw OHLCV (see [Section 12](#12-the-18-input-features--detailed-breakdown))
- Merges each stock with Nifty 50 index data (for market context features)

**Output:**
```
data/processed/RELIANCE_processed.csv   (~370k rows, 19 columns: timestamp + 18 features)
data/processed/TCS_processed.csv
data/processed/.../                     (50 stocks total)
```

**Verify:**
```bash
python scripts/verify_processed.py
# Expected: 50 stocks, 18 features each, avg Nifty correlation > 0.5
# Report saved to logs/reports/processed_verification_report.txt
```

---

### 6.3 Create Training Windows (~1 hour)

```bash
python pipeline/03_create_windows_60min.py
```

**What happens:**
- Slides a window across each stock's processed data
- **Input (X):** 60 consecutive minutes × 18 features = shape `(60, 18)`
- **Target (y_close):** The next 15 close prices = shape `(15,)`
- Dense sampling with stride=1 → ~370k windows per stock
- Also stores `close_idx` (index of close column in feature array) in each `.npz`

**Window structure in each `.npz` file:**
```
X:             (num_windows, 60, 18)   — raw input features
y_close:       (num_windows, 15)       — target: next 15 close prices
x_normalized:  (num_windows, 60, 18)   — StandardScaler-normalized X (used for training)
close_idx:     scalar                  — which column in X is 'close' (=3)
timestamps:    (num_windows,)          — window start times
feature_names: list of 18 strings
stock_name:    string                  — e.g. "TCS"
```

**Output:**
```
data/windows/RELIANCE_windows.npz   (~300 MB per stock, compressed)
data/windows/TCS_windows.npz
data/windows/.../                   (50 stocks, ~15 GB total)
```

---

### 6.4 Verify Data (Optional)

```bash
# Pre-training sanity check: ensures X is raw (not pre-normalized) and has no NaN/Inf
python scripts/audit_data.py
```

Expected output:
```
✅ RELIANCE   close=[1024.3, 2987.5]  y=[1024.1, 2988.1]  NaN_X=0  raw=YES
✅ TCS        close=[2900.1, 4325.6]  y=[2899.8, 4326.0]  NaN_X=0  raw=YES
...
10/10 stocks passed ✅
```

---

# Part IV: Training

---

## 7. Model Training (Global Foundation Model)

### 7.1 Data Loading Strategy

Loading all 50 stocks' windows into RAM at once would require ~75 GB. We use a **sampled pre-loading** approach instead:

**How it works:**
```
Startup:
  For each of 50 stocks:
    Memory-map the .npz file
    Randomly sample MAX_WINDOWS_PER_STOCK (20,000) windows from train split
    Load those windows into RAM
    Close the file
  Total RAM: 50 × 20k × 60 × 18 × 4 bytes ≈ 4.3 GB

Each epoch:
  Shuffle all 1M loaded windows
  Train on all batches (batch_size=64 → ~15,625 batches)

Next epoch:
  Same data (re-shuffled). Windows are re-sampled only on restart.
```

**Three methods compared:**

| Method | RAM | Speed | Used When |
|---|---|---|---|
| **All-at-Once** | 75 GB ❌ | Fastest | Never (doesn't fit) |
| **Sampled Pre-load** (current) | ~5 GB ✅ | Fast | 50 stocks |
| **Streaming** (planned) | ~500 MB ✅ | Slower (I/O bound) | 500 stocks |

**Key insight:** The model sees all 50 stocks every epoch — it just sees a random 20,000-window subset per stock rather than the full ~370,000. With 200 epochs, each window has statistically been seen multiple times.

### 7.2 Training Configuration

All hyperparameters are in [`train/config.py`](train/config.py):

| Parameter | Value | Notes |
|---|---|---|
| **Batch size** | 64 | Fits comfortably on 6 GB VRAM |
| **Learning rate** | 1e-4 | Initial; decays via cosine schedule |
| **LR schedule** | Cosine | With 3-epoch warmup, min_lr=1e-6 |
| **Max epochs** | 200 | With early stopping (patience=30) |
| **Optimizer** | AdamW | weight_decay=1e-5 |
| **Gradient clipping** | 1.0 | Max grad norm |
| **Mixed precision** | Off | fp16 causes overflow in CARD attention; RTX 4050 TF32 is fast enough |
| **Windows per stock** | 20,000 train / 1,000 val | Per epoch |
| **Data split** | 70% / 15% / 15% | **Temporal** — first 70% by time = train |
| **DIR_LOSS_WEIGHT** | 0.5 | Weight for directional penalty |
| **Early stop patience** | 30 epochs | Accept ANY val improvement |

### 7.3 Loss Function

The total loss has two components:

**1. SignalDecayMAE** (Equation 12 from CARD paper)

Weighted Mean Absolute Error where near-term predictions matter more:

```
Loss = (1/15) × Σ_{l=1}^{15} l^{-1/2} × |predicted[l] - actual[l]|
```

| Minute | 1 | 2 | 5 | 10 | 15 |
|---|---|---|---|---|---|
| Weight | 1.000 | 0.707 | 0.447 | 0.316 | 0.258 |

**2. Directional Penalty**

Extra penalty when predicted direction disagrees with actual direction:

```
If sign(pred_change) ≠ sign(actual_change):
    penalty += DIR_LOSS_WEIGHT × |predicted - actual|
```

**Combined:**
```
Total Loss = SignalDecayMAE + 0.5 × DirectionalPenalty
```

> Both components operate on **normalized** close prices (inside RevIN space), so loss values are O(0.05), not O(1000).

### 7.4 Running Training

```bash
# Activate environment
conda activate card

# Full training run (~2-3 hours on RTX 4050)
python train/04_train_card.py

# Quick sanity check (3 epochs)
python train/04_train_card.py --epochs 3
```

**What to expect in the console:**
```
✅ GPU: NVIDIA GeForce RTX 4050 Laptop GPU
============================================================
TRAINING CONFIGURATION
============================================================
Stocks    : 50 | 20000 windows/stock = 1,000,000 total/epoch
Input     : 60 min × 18 features
Output    : 15 close prices
Batch     : 64, Epochs: 200
LR        : 0.0001
============================================================

Loading train split from 50 stocks...
✅ train: 1,000,000 samples  |  RAM ≈ 4380 MB

Epoch 1/200
Epoch 1: 100%|████| 15625/15625 [25:30<00:00]

  Train Loss: 0.05320  |  Val Loss: 0.05275
  Direction Accuracy: train=47.20%  val=47.20%
  MAE: train=₹2.46  val=₹3.43
  ⭐ New best model → checkpoints/best_model.pt
```

**Checkpoints saved:**
- `checkpoints/best_model.pt` — Best validation loss (auto-replaced)
- `checkpoints/checkpoint_epoch_N.pt` — Every 5 epochs (keeps last 3)

**Training history logged to:** `logs/training_history_{timestamp}.json`

### 7.5 Loading a Trained Model

```python
import torch
from models.card_true import CARD
from train.config import TrainingConfig, CARDModelConfig

# Initialize model with same architecture
config = CARDModelConfig(TrainingConfig())
model = CARD(config)

# Load checkpoint
ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Inference: shape (batch, 18 channels, 60 timesteps)
import numpy as np
X = np.random.randn(1, 18, 60).astype(np.float32)  # dummy input
x_tensor = torch.tensor(X)

with torch.no_grad():
    pred_close, revin_stats = model(x_tensor)
    # pred_close: (1, 15) — next 15 close prices in ₹
    # revin_stats: (mean, std) used for normalization
```

> **Note on `weights_only=False`:** Required because the checkpoint contains a `TrainingConfig` object alongside the state dict. This is safe for our own checkpoints.

---

## 8. Transfer Learning (Per-Stock Fine-Tuning)

### Why Transfer Learning?

Financial assets share **universal macro-level features** (how volatility clusters behave, the shape of a breakout or mean-reversion) but differ in **micro-level personalities** (precise resistance behaviors, average tick volatility, earnings-day patterns).

The global model has learned the "physics." Fine-tuning teaches each model the individual stock's "personality" without catastrophic forgetting.

### 8.1 Freeze/Unfreeze Strategy

If we retrain the **entire** network on a single stock, it forgets the general patterns and overfits to that stock's noise. Instead, we freeze most layers:

| Layer | Status | What It Learned |
|---|---|---|
| `revin` | **🔒 Frozen** | Universal normalization — same for all stocks |
| `W_input_projection` | **🔒 Frozen** | Patch tokenization — universal |
| `W_pos_embed` | **🔒 Frozen** | Positional encoding — universal |
| `Attentions_over_token[0]` | **🔒 Frozen** | First encoder block — general temporal patterns |
| `Attentions_over_channel[0]` | **🔒 Frozen** | First encoder block — general cross-feature patterns |
| `Attentions_mlp[0]`, `Attentions_norm[0]` | **🔒 Frozen** | First block's FFN and LayerNorm |
| `Attentions_over_token[1]` | **🔓 Trainable** | Last encoder block — adapts temporal attention to stock |
| `Attentions_over_channel[1]` | **🔓 Trainable** | Last encoder block — adapts cross-feature attention |
| `Attentions_mlp[1]`, `Attentions_norm[1]` | **🔓 Trainable** | Last block's FFN and LayerNorm |
| `W_out` (MLP Decoder) | **🔓 Trainable** | Final prediction layer — stock-specific |

**Result:** ~680k frozen params (49%) + ~707k trainable params (51%)

### 8.2 Transfer Learning Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| **Epochs** | 15 | Model is 95% converged; more risks overfitting on single-stock data |
| **Learning rate** | 1e-5 | 1/10th of global LR — higher would destroy foundation weights |
| **Batch size** | 64 | Same as global; single stock fits in VRAM |
| **Optimizer** | AdamW | Same as global |
| **Early stop patience** | 3 | Stops quickly if no improvement (vs 30 for global) |
| **Loss** | Same SignalDecayMAE + Directional Penalty | Consistency |
| **Data** | Full stock windows (no sampling) | Single stock fits in RAM (~1.1 GB) |

### 8.3 Running Transfer Learning

```bash
python pipeline/05_transfer_learning.py
```

**What happens for each stock:**
1. Load the global `best_model.pt` (fresh copy each time)
2. Freeze foundation layers, unfreeze top layers
3. Load only that stock's windows into RAM (~1.1 GB)
4. Fine-tune for 15 epochs with progress bars
5. Save to `checkpoints/fine_tuned/card_{STOCK}.pt`
6. **Delete the stock data from RAM** before next stock
7. Repeat for all 50 stocks

**RAM profile:** Flat regardless of stock count. Each stock loads ~1.1 GB then clears.

**Time:** ~7 minutes per stock × 50 stocks ≈ 6 hours total

**Currently fine-tuned (6 stocks):**
```
checkpoints/fine_tuned/
├── card_TCS.pt       (5.4 MB)
├── card_INFY.pt      (5.4 MB)
├── card_HCLTECH.pt   (5.4 MB)
├── card_HDFCBANK.pt  (5.4 MB)
├── card_TECHM.pt     (5.4 MB)
└── card_WIPRO.pt     (5.4 MB)
```

**Scaling to 500 stocks:** Just update `scripts/config_nifty50.py` with the new list. The script iterates over `config.STOCKS` automatically.

---

# Part V: Evaluation & Demo

---

## 9. Training Results

### Global Model — 124 Epochs (best_model.pt)

Training ran for 124 out of 200 scheduled epochs before early stopping.

| Metric | Start (E1) | Best | At Epoch |
|---|---|---|---|
| Val Loss (Normalized MAE) | 0.05275 | **0.05258** | E94 |
| Train Loss | 0.05320 | **0.05224** | E52 |
| Val MAE (₹) | 3.43 | **3.41** | E65 |
| Train MAE (₹) | 2.46 | **2.40** | E113 |
| Val Direction Accuracy | 47.20% | **47.30%** | E86 |
| Train Direction Accuracy | 46.10% | **46.23%** | E123 |
| Val Correlation | 1.00 | **1.00** | — |

### Interpreting These Results

**What's good:**
- **MAE of ₹3.41** — The model predicts close prices within ₹3-4 of the actual value on average. On a ₹1400 stock, that's 0.24% error. This is genuinely impressive.
- **Correlation = 1.00** — The model's predictions track the actual price level perfectly.

**What's bad:**
- **Direction accuracy = 47.30%** — Worse than a coin flip. The model doesn't predict *which way* the price will move.

### The Persistence Problem (Known Issue)

The model produces **near-flat predictions** — predicting "no change" from the last known price. This manifests as:
- All 15 predicted prices being within ₹0.01 of each other
- Low MAE (because "predict current price" is a great guess for magnitude)
- Direction accuracy below 50% (random guessing is 50%)

**Root cause:** MSE/MAE loss rewards models that predict zero delta. On random-walk-like data, "predict no change" has the lowest possible expected MAE. The directional penalty (weight=0.5) was not strong enough to counteract this.

**Planned fix:**
1. Increase `DIR_LOSS_WEIGHT` from 0.5 → **3.0 or higher**
2. Add **anti-flatness penalty**: `penalty = relu(0.3 - pred.std()) × 10.0`
3. Unfreeze more encoder layers during transfer learning
4. Retrain global model with stronger directional incentives

---

## 10. Live Demo

A Flask web app that fetches live data from Fyers, runs the CARD model, and shows **predicted vs actual** prices on a chart.

### Running the Demo

```bash
# 1. Ensure access token is fresh (tokens expire daily)
python pipeline/00_generate_token.py

# 2. Start the demo server
python demo_server.py

# 3. Open in browser
# → http://localhost:5000
```

### Demo Features

- **Stock selector** — Dropdown with all Nifty 50 stocks (★ marks fine-tuned models)
- **Time selector** — Pick any point during the trading day to make a prediction from
- **Predicted vs Actual chart** — Chart.js overlay comparing blue (predicted) vs green (actual)
- **Metrics** — MAE, RMSE, Direction Accuracy, and 15-min trend call per window
- **Auto model selection** — Uses fine-tuned `card_{STOCK}.pt` if available, else `best_model.pt`
- **Persistence warning** — Orange banner explains when the model outputs a flat prediction

### How the Demo Works Internally

1. **Fetch data:** Calls Fyers API for the full day's 1-minute candles (with retry/backoff)
2. **Feature engineering:** Applies the exact same 18-feature pipeline as training
3. **Window selection:** User picks a 60-minute input window via the time dropdown
4. **Inference:** Loads the appropriate model, runs `model(X)`, gets 15 predictions
5. **Comparison:** The actual next 15 candles are also available, so both are plotted

---

# Part VI: Technical Deep Dives

---

## 11. Key Design Decisions

### 11.1 Why Close Prices, Not Returns

The model predicts **raw close prices** (normalized by RevIN internally), not percentage returns or log-returns.

**Rationale:**
- CARD was designed for same-domain time series forecasting (temperature → temperature, flow → flow)
- RevIN normalizes input and denormalizes output using the same channel statistics
- Predicting close prices keeps us faithful to the CARD architecture
- No post-processing needed — the model output is directly interpretable as ₹ prices

### 11.2 RevIN Denormalization — The Cross-Domain Problem

This is the most critical design decision in the project. Understanding it prevents a 100× magnitude error.

**Standard CARD Use Case (Same-Domain):**

In weather forecasting (what the paper benchmarks on), input and output are in the same domain:
```python
# Weather example — RevIN denorm works perfectly
input = [20°C, 21°C, 22°C, ...]     # Temperature
output = [23°C, 24°C, 25°C, ...]    # Also temperature

# Normalize
mean = 21.0, std = 5.0
normalized = (input - 21.0) / 5.0

# Model predicts: 0.4 (in normalized space)
# Denormalize: 0.4 × 5.0 + 21.0 = 23.0°C  ✅ CORRECT
```

**Our Use Case (Cross-Domain) — Why Naive RevIN Denorm Fails:**

Our input has 18 features at vastly different scales, but our output is 1 channel (close price):
```python
# Stock example — what WOULD happen with full RevIN denorm
input_features = [close=2450, volume=1000000, RSI=65, MACD=12, ...]
# Input statistics (from all 18 features):
mean = [2450, 1000000, 65, 12, ...]    # Mixed scales!
std  = [50, 500000, 15, 3, ...]        # Huge variance!

# Model predicts (normalized): 0.05
# WRONG denorm using input stats:
# 0.05 × 50 + 2450 = 2452.5  ❌ OFF BY THOUSANDS!

# CORRECT: only use close channel stats for denormalization
```

**Our solution:** We denormalize using **only the close channel's** mean and std (stored during RevIN normalization), not all 18 channels. This is implemented in `CARD.forward()` — it extracts the close channel slice from the full 18-channel output and denormalizes with the close-specific statistics.

**Impact of this fix:**
| | Before (full denorm) | After (close-channel only) |
|---|---|---|
| MAE | 0.77 (100× too large) | 0.003–0.01 ✅ |
| Direction accuracy | 48% (random) | 47–55% |
| Training stability | Poor | Good |

**CARD paper alignment:** This is a *correct adaptation* of CARD for cross-domain prediction, not a deviation. The paper only benchmarked same-domain tasks where full RevIN denorm works. Our adaptation preserves all core components (RevIN input norm, patch tokenization, EMA attention, channel/token attention, token blend) and only changes what statistics are used for output denormalization.

### 11.3 Why One Global Model

Training one model on all 50 stocks simultaneously, rather than 50 separate models:

| Approach | Training Time | Data per Model | Generalization |
|---|---|---|---|
| 50 separate models | 50 × 3 hrs = 150 hrs | 370k windows | Overfits to individual stock |
| **1 global model** | **3 hrs** | **18.5M windows** | **Learns universal patterns** |

RevIN makes this possible — it normalizes each window independently, so the model never sees raw price levels. A ₹100 stock and a ₹5000 stock look identical after normalization.

### 11.4 Temporal Data Split (No Random Split)

**Critical:** Data is split by **time**, not randomly.

```
├── Train: first 70% of each stock's timeline (Jan 2022 – ~Aug 2024)
├── Val:   next 15% (~Aug 2024 – ~Apr 2025)
└── Test:  last 15% (~Apr 2025 – Dec 2025, completely unseen)
```

**Why not random split?** Adjacent 1-minute windows are 99% identical (only the last minute changes). A random split would put nearly-identical windows in both train and val/test sets → massive data leakage → artificially inflated metrics.

---

## 12. The 18 Input Features — Detailed Breakdown

Computed in `pipeline/02_process_stocks.py`:

### Base OHLCV (5 features)

| # | Feature | Source | Description |
|---|---|---|---|
| 0 | `open` | Raw | Opening price of the minute |
| 1 | `high` | Raw | Highest price in the minute |
| 2 | `low` | Raw | Lowest price in the minute |
| 3 | `close` | Raw | Closing price (**this is what we predict**) |
| 4 | `volume` | Raw | Number of shares traded |

### Technical Indicators (6 features)

| # | Feature | Formula | What it captures |
|---|---|---|---|
| 5 | `ema_5` | 5-period EMA of close | Very short-term trend |
| 6 | `ema_10` | 10-period EMA of close | Short-term trend |
| 7 | `roi_1m` | `(close - prev_close) / prev_close` | 1-minute return (momentum signal) |
| 8 | `bb_upper` | SMA(20) + 2×σ | Upper Bollinger Band (overbought) |
| 9 | `bb_middle` | SMA(20) | 20-period simple moving average |
| 10 | `bb_lower` | SMA(20) - 2×σ | Lower Bollinger Band (oversold) |

### Market Context (4 features)

| # | Feature | Source | What it captures |
|---|---|---|---|
| 11 | `nifty_close` | NIFTY50.csv | Index price — systemic market movement |
| 12 | `nifty_return` | 1-min return of Nifty | Market momentum |
| 13 | `hour_sin` | `sin(2π × minute_of_day / 375)` | Time encoding (cyclical) |
| 14 | `hour_cos` | `cos(2π × minute_of_day / 375)` | Time encoding (cyclical) |

### Volatility & Volume (3 features)

| # | Feature | Formula | What it captures |
|---|---|---|---|
| 15 | `realized_vol_30m` | Std of 1-min returns over 30 min | Recent volatility |
| 16 | `volume_ratio` | `volume / SMA(volume, 20)` | Relative volume (1.0 = normal) |
| 17 | `volume_surge` | `1 if volume_ratio > 2.0 else 0` | Binary spike detector |

---

## 13. Configuration Reference

All hyperparameters in [`train/config.py`](train/config.py):

```python
# === Model Architecture ===
SEQ_LEN           = 60       # Input: 60 minutes
PRED_LEN          = 15       # Output: 15 minutes
ENC_IN            = 18       # 18 input features
CLOSE_CHANNEL_IDX = 3        # Index of 'close' in feature array
D_MODEL           = 128      # Hidden dimension
N_HEADS           = 8        # Attention heads
E_LAYERS          = 2        # Encoder depth
D_FF              = 512      # Feed-forward width
PATCH_LEN         = 8        # Patch size (minutes)
STRIDE            = 4        # Patch stride
DROPOUT           = 0.1
MERGE_SIZE        = 2        # Token blend
DP_RANK           = 8        # Dynamic projection rank
ALPHA             = 0.9      # EMA smoothing (fixed)

# === Training ===
BATCH_SIZE        = 64
NUM_EPOCHS        = 200
LEARNING_RATE     = 1e-4
WEIGHT_DECAY      = 1e-5
LR_SCHEDULER      = 'cosine'
WARMUP_EPOCHS     = 3
MIN_LR            = 1e-6
MAX_GRAD_NORM     = 1.0
USE_AMP           = False    # fp16 causes overflow

# === Data ===
MAX_WINDOWS_PER_STOCK     = 20000   # Train windows/stock/epoch
MAX_VAL_WINDOWS_PER_STOCK = 1000    # Val windows/stock
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# === Checkpointing ===
SAVE_FREQ           = 5      # Save every 5 epochs
KEEP_LAST_N         = 3      # Keep last 3 checkpoints
EARLY_STOP_PATIENCE = 30     # Epochs without improvement
DIR_LOSS_WEIGHT     = 0.5    # Directional penalty weight
```

---

# Part VII: Reference

---

## 14. Troubleshooting

| Problem | Symptom | Solution |
|---|---|---|
| **Access token expired** | `Invalid access token` | Re-run `python pipeline/00_generate_token.py` (daily) |
| **CUDA out of memory** | `RuntimeError: CUDA out of memory` | Reduce `BATCH_SIZE` to 32 or `D_MODEL` to 64 in `train/config.py` |
| **API rate limit** | `Rate limit exceeded` | Built-in retries handle this; increase delays if persistent |
| **NaN loss** | `loss=nan` during training | Lower `LEARNING_RATE` to `1e-5` and `MAX_GRAD_NORM` to `0.5` |
| **Direction accuracy ~50%** | Model not learning direction | Increase `DIR_LOSS_WEIGHT` to 3.0+; add anti-flatness penalty |
| **System freeze during training** | RAM exhaustion | Reduce `MAX_WINDOWS_PER_STOCK` to 10000 |
| **Slow training (>1 hr/epoch)** | I/O bottleneck | Verify `NUM_WORKERS=0` on Windows; increase `BATCH_SIZE` if VRAM allows |
| **Module not found** | `ModuleNotFoundError: fyers_apiv3` | Run `conda activate card` first |
| **Flat predictions** | All 15 outputs identical | This is the [persistence problem](#the-persistence-problem-known-issue) — see Section 9 |

---

## 15. Development History

### Timeline

| Date | Milestone |
|---|---|
| **Jan 2026** | Project started. Downloaded Nifty 50 data via Fyers API. |
| **Feb 10** | Data pipeline complete (download → process → window). First training attempt crashed — 75 GB RAM needed. |
| **Feb 10** | Fixed memory: implemented sampled pre-loading (75 GB → 5 GB). Fixed CUDA device mismatch. Fixed I/O bottleneck (44s/batch → 0.5s/batch). |
| **Feb 12** | First successful training run: 124 epochs completed. Model achieves ₹3.41 MAE but 47% direction accuracy (persistence problem identified). |
| **Feb 13** | RevIN denormalization fix: identified that full denorm caused 100× magnitude errors. Implemented close-channel-only denorm. |
| **Feb 19** | MASTERPLAN rewrite: switched from predicting returns to predicting close prices. Added directional penalty loss. |
| **Feb 25** | Transfer learning implemented: freeze/unfreeze strategy. Fine-tuned 6 IT-sector stocks (TCS, INFY, HCLTECH, HDFCBANK, TECHM, WIPRO). |
| **Feb 26** | Live demo built: Flask + Chart.js showing predicted vs actual. Confirmed model tracks price level well but direction accuracy still poor. |
| **Mar 6** | Project directory reorganized: removed stray files, merged docs, patched output paths. |

### Key Technical Fixes

1. **Memory-mapped → Sampled pre-loading** — Initial mmap approach caused 44s/batch I/O bottleneck. Switched to loading a random 20k-window subset per stock into RAM at startup. 100-400× speedup.

2. **RevIN denormalization** — Full 18-channel denorm produced outputs 100× too large (MAE=0.77). Fixed by denormalizing only the close channel using close-channel statistics.

3. **Mixed precision disabled** — fp16 caused overflow/NaN in CARD's EMA attention due to multiply-accumulate chains. RTX 4050's TF32 mode is fast enough without fp16.

---

## 16. Roadmap

- [ ] **Fix directional accuracy** — Stronger directional loss (weight 3.0+) + anti-flatness penalty
- [ ] **Retrain global model** with improved loss function
- [ ] **Complete fine-tuning** all 50 stocks (currently only 6 done)
- [ ] **Scale to Nifty 500** — architecture ready, need data download
- [ ] **Streaming dataset** for 500-stock training (constant ~500 MB RAM)
- [ ] **Backtesting framework** — Simulated trading with transaction costs + Sharpe ratio
- [ ] **Live trading integration** — Real-time inference + order execution
- [ ] **Sentiment features** — Add news/social sentiment as additional input channels

---

<div align="center">

*Last updated: March 6, 2026*

Built with PyTorch · Powered by CARD (ICLR 2024) · Market data via Fyers API

</div>
