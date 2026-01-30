# CARD Stock Prediction - Nifty 50

Deep learning-based stock price prediction using **Channel Aligned Robust Blend Transformer (CARD)** architecture for Nifty 50 stocks.

## 🎯 Project Overview

**Objective:** Predict 15-minute ahead stock prices using 60-minute historical data with true CARD architecture.

**Key Features:**
- ✅ 60-minute input window (60 timesteps @ 1-min frequency)
- ✅ 15-minute prediction horizon  
- ✅ True CARD implementation (patching + dual attention)
- ✅ Multi-task learning (returns + volatility)
- ✅ Single base model trained on all 50 Nifty stocks
- ✅ Real-time prediction capability

## 📊 Expected Performance

- **Direction Accuracy:** 54-58% (vs. 50% random)
- **MAE:** 0.00085-0.00095 (15-min ahead)
- **Correlation:** 0.35-0.50
- **Sharpe Ratio:** 0.8-1.2

## 🏗️ Architecture

```
Input (60 min × 18 features)
    ↓
[Patch Embedding] (8-min patches, stride=4 → 13 tokens)
    ↓
[Channel Attention] (attend across 18 features)
    ↓
[Token Attention] (attend across 13 time tokens)
    ↓
[Token Blend] (multi-scale aggregation)
    ↓
[Multi-task Head] → Returns (15 steps) + Volatility (15 steps)
```

## 📁 Project Structure

```
card-stock-prediction/
├── data/
│   ├── raw/                    # Raw OHLCV downloads
│   ├── market_indices/         # Nifty50 index data
│   ├── processed/              # Cleaned + 18 features
│   ├── combined/               # All stocks merged
│   └── windows/                # Windowed training data
├── models/
│   ├── card_stock.py           # Main CARD architecture
│   ├── channel_attention.py    # Channel-aligned attention
│   ├── token_attention.py      # Token attention
│   ├── token_blend.py          # Multi-scale blending
│   ├── revin.py                # Reversible Instance Norm
│   └── checkpoints/            # Saved model weights
├── losses/
│   ├── signal_decay_loss.py    # Signal decay MAE
│   └── multi_task_loss.py      # Returns + Volatility loss
├── scripts/
│   ├── 01_download_nifty50.py       # Download 50 stocks + index
│   ├── 02_process_stocks.py         # Add 18 features
│   ├── 03_create_windows_60min.py   # Create training windows
│   ├── 04_train_card_base.py        # Train base model
│   └── 05_evaluate.py               # Test set evaluation
└── results/
    ├── training_curves/
    ├── predictions/
    └── metrics/
```

## 🔧 Setup Instructions

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd card-stock-prediction
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n card-stock python=3.10
conda activate card-stock

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Fyers API Credentials

Create `.env` file in project root:
```bash
CLIENT_ID=your_fyers_client_id
SECRET_KEY=your_fyers_secret_key
ACCESS_TOKEN=your_fyers_access_token
```

**Generate access token:**
```bash
python scripts/00_generate_token.py
```

### 5. Create Directory Structure
```bash
mkdir -p data/{raw,market_indices,processed,combined,windows}
mkdir -p models/checkpoints
mkdir -p results/{training_curves,predictions,metrics}
mkdir -p logs
```

## 🚀 Usage

### Phase 1: Download Data (Nifty 50 stocks)
```bash
python scripts/01_download_nifty50.py
```
**Output:** 50 stock CSVs + Nifty index in `data/raw/`

### Phase 2: Process & Add Features
```bash
python scripts/02_process_stocks.py
```
**Output:** Processed CSVs with 18 features in `data/processed/`

### Phase 3: Create 60-Min Windows
```bash
python scripts/03_create_windows_60min.py
```
**Output:** Training/Val/Test numpy arrays in `data/windows/`

### Phase 4: Train CARD Model
```bash
python scripts/04_train_card_base.py
```
**Output:** Trained model checkpoint in `models/checkpoints/`

### Phase 5: Evaluate
```bash
python scripts/05_evaluate.py
```
**Output:** Metrics and visualizations in `results/`

## 📊 Features (18 Total)

### OHLCV (5)
- Open, High, Low, Close, Volume

### Technical Indicators (6)
- EMA_5, EMA_10
- ROI_1m (1-minute return)
- BB_upper, BB_middle, BB_lower (Bollinger Bands)

### Market Context (4)
- Nifty_close (index price)
- Nifty_return (index return)
- Hour_sin, Hour_cos (time of day, cyclic)

### Volatility & Volume (3)
- Realized_vol_30m (30-min rolling volatility)
- Volume_ratio (vs. 20-min average)
- Volume_surge (boolean indicator)

## 🎓 Model Specifications

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Input Length | 60 minutes | CARD needs ≥96, 60 is compromise |
| Prediction Horizon | 15 minutes | Intraday trading relevant |
| Patch Length | 8 | Creates 13 tokens |
| Patch Stride | 4 | Overlapping patches |
| Hidden Dimension | 128 | Balance capacity/speed |
| Attention Heads | 8 | Multi-head attention |
| Encoder Layers | 2 | Token + Channel attention |
| Dropout | 0.1 | Regularization |
| Batch Size | 64 | Fits 6GB VRAM |
| Learning Rate | 1e-3 → 1e-6 | Cosine annealing |
| Epochs | 100 | Early stopping (patience=15) |

## 📈 Training Details

**Data Split (Time-based):**
- Train: 70% (2022-01 to ~2024-04)
- Val: 15% (~2024-04 to ~2024-10)
- Test: 15% (~2024-10 to 2025-12)

**Loss Function:**
- Signal Decay MAE for returns (weights near-term higher)
- MSE for volatility
- Combined: `0.7 × returns_loss + 0.3 × vol_loss`

**Optimizer:**
- AdamW with weight decay 1e-5
- Gradient clipping (max_norm=1.0)
- Cosine annealing with warm restarts

**Training Time:**
- Expected: ~6-8 hours on RTX 4050
- Checkpoint: Every 10 epochs
- Early stopping: Patience 15

## 📊 Evaluation Metrics

### Returns Prediction
- MAE per timestep (1-15 min ahead)
- Direction accuracy (% correct up/down)
- Correlation (predicted vs. actual)

### Volatility Prediction
- MSE
- Coverage (% actual within predicted bands)

### Trading Simulation
- Sharpe ratio
- Max drawdown
- Win rate

## 🔬 Ablation Studies (Optional)

Compare:
- [x] CARD (full) vs. CARD (no channel attention)
- [x] CARD vs. GRU baseline
- [x] CARD vs. PatchTST
- [x] Signal decay loss vs. plain MSE

## 📚 References

1. **CARD Paper:** [Channel Aligned Robust Blend Transformer for Time Series Forecasting](https://arxiv.org/abs/2305.12095)
2. **PatchTST:** [A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730)
3. **RevIN:** [Reversible Instance Normalization](https://arxiv.org/abs/2105.14428)

## ⚠️ Important Notes

### Data Quality
- Nifty 50 stocks are highly liquid (minimal missing data)
- Market hours: 9:15 AM - 3:30 PM IST
- Predictions start at 10:15 AM (need 60-min history)

### GPU Memory
- 6GB VRAM is sufficient with batch_size=64
- If OOM: reduce to batch_size=32
- Training uses ~4.5GB VRAM peak

### Known Limitations
- Predicting minute-level stock prices is inherently noisy
- 54-58% direction accuracy is realistic ceiling
- Model cannot predict black swan events

## 🐛 Troubleshooting

**Issue:** Fyers API rate limit exceeded  
**Solution:** Add `time.sleep(1)` between stock downloads

**Issue:** CUDA out of memory  
**Solution:** Reduce `batch_size` in training script

**Issue:** Missing data for some stocks  
**Solution:** Check `data/logs/download_errors.txt`

**Issue:** Training loss not decreasing  
**Solution:** Check learning rate, try reducing to 5e-4

## 📞 Contact

For questions or issues, please open a GitHub issue.

## 📄 License

MIT License - See LICENSE file for details

---

**Last Updated:** January 2026  
**Python Version:** 3.10+  
**PyTorch Version:** 2.1.0
