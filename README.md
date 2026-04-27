# CARD: Channel Aligned Robust Blend Transformer

> A production-grade quantitative ML pipeline for intraday return prediction on the NSE Nifty 500 universe, based on the [ICLR 2024 CARD paper](https://openreview.net/forum?id=MJksrOhurza).

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch) ![License](https://img.shields.io/badge/License-MIT-green)

---

## What This Project Does

CARD predicts **15-minute forward returns** for Indian equities using a 81-channel feature suite and a Transformer-based architecture with dual attention (channel + token). The system is engineered to handle the full research lifecycle:

- **Data Acquisition** — Fyers API → NSE Nifty 500 1-minute OHLCV data
- **Feature Engineering** — 81 channels: momentum, volatility, volume, cross-asset, time regime
- **Memory-Efficient Windowing** — Index-based slicing that reduces storage from **1.3 TB → ~2 GB**
- **Training Engine** — CUDA-optimized training loop with TensorBoard + W&B logging
- **Modular Architecture** — Strict Library → Factory → Trigger separation for research reproducibility

---

## Architecture

The project follows a **Library-Factory-Trigger** pattern enforced by architectural rules in `.agents/rules/`:

```
CARD_FOR_TRADING/
├── src/                    # Core library (models, engine, features, utils)
│   ├── models/             # CARD Transformer, attention, layers, loss
│   ├── engine/             # Training loop, LR scheduler
│   ├── features/           # 81-channel feature engineering engine
│   ├── data/               # PyTorch Dataset, window builder, downloader
│   └── utils/              # Fyers auth, multi-backend logging
├── pipeline/               # Data factory (thin wrappers over src/)
│   ├── 00_auth.py          # Fyers API token generation
│   ├── 01_download.py      # OHLCV data download
│   ├── 02_process.py       # Feature engineering
│   └── 03_windowing.py     # Sliding window creation
├── data/
│   └── configs/            # NSE Nifty 500 stock universe definition
├── docs/                   # System map, development history, project log
├── archive/                # Preserved legacy training scripts
└── vendor/pandas_ta/       # Custom pandas_ta fork (install separately)
```

> See [docs/SYSTEM_MAP.md](docs/SYSTEM_MAP.md) for a full file-by-file responsibility breakdown and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the complete Phase 0–5 research history.

---

## Quick Start (New Clone)

### Prerequisites

- **Conda** (Anaconda or Miniconda)
- **CUDA-capable GPU** (tested on RTX 4050 6GB)
- **Fyers API account** — required for data download ([register here](https://myapi.fyers.in))

### 1. Clone and Set Up Environment

```powershell
git clone https://github.com/Nitin-Nandan/CARD_FOR_TRADING.git
cd CARD_FOR_TRADING

# Create and activate the conda environment
conda create -n card python=3.10 -y
conda activate card

# Install PyTorch with CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install all project dependencies
pip install -r requirements.txt
```

### 2. Install the Vendor `pandas_ta` Fork

The project uses a custom fork of `pandas_ta` (v0.3.14b, Python 3.10 compatible). The source is excluded from git — you must download and install it manually:

```powershell
# Option A: Install directly from the released fork
pip install git+https://github.com/twopirllc/pandas-ta.git@v0.3.14b

# Option B: If you have the vendor/ folder populated
conda activate card
pip install -e .\vendor\pandas_ta
```

### 3. Configure Credentials

```powershell
# Copy the template and fill in your Fyers API credentials
Copy-Item .env.template .env
# Edit .env and add: FYERS_APP_ID, FYERS_SECRET, FYERS_REDIRECT_URI
```

---

## Data Regeneration

> [!IMPORTANT]
> `data/raw/`, `data/processed/`, `data/windows/`, and `checkpoints/` are **not included in this repository** (they contain TB-scale binary data). You must regenerate them after cloning.

Run the 4-step pipeline **in order**:

```powershell
conda activate card

# Step 0: Authenticate with Fyers API (opens browser for token)
python pipeline/00_auth.py

# Step 1: Download 1-minute OHLCV for Nifty 500 stocks (~hours, ~20GB raw CSVs)
python pipeline/01_download.py

# Step 2: Compute 81-channel features (outputs Parquet files to data/processed/)
python pipeline/02_process.py

# Step 3: Build index-based sliding windows (outputs .npz to data/windows/, ~2GB)
python pipeline/03_windowing.py
```

Each step is a thin wrapper over `src/` — inspect the pipeline scripts for available CLI flags (e.g., `--stocks`, `--start-date`, `--end-date`).

---

## Model Checkpoints

> [!NOTE]
> Trained model weights (`.pt` files) are gitignored. To obtain pre-trained weights, contact the repository maintainer, or train from scratch using the engine in `src/engine/trainer.py`.

To resume training or run inference, place `.pt` files in the `checkpoints/` directory.

---

## Technical Highlights

| Component | Detail |
|---|---|
| **Architecture** | CARD Transformer: RevIN → ConvNeXtV2 → Multi-Scale Temporal Attention → Dual heads |
| **Feature Suite** | 81 channels: RSI, MACD, ATR, OBV, MFI, VWAP, Parkinson vol, beta vs NIFTY 50, time regimes |
| **Loss Function** | `CombinedReturnLoss`: Signal-decay MAE (1000x scaled) + directional sign penalty |
| **Data Efficiency** | Index-based windowing: 1.3 TB → ~2 GB; SSD mmap streaming with ~0 MB RAM overhead |
| **Input / Output** | `(B, 81, 60)` — 60 one-minute bars across 81 channels → 15-step return forecast |
| **Best Result** | 49.7% directional accuracy (Phase 4 ceiling; Phase 5 targets alpha via regime labels) |

---

## Development Status

| Phase | Description | Status |
|---|---|---|
| 0 | Solved 1.3 TB storage problem with index-based windowing | ✅ |
| 1 | 18-channel OHLCV + basic indicators | ✅ |
| 2 | Pivoted from price to **returns prediction** (stationarity) | ✅ |
| 3 | Expanded to **81 channels** with volatility/trend regime detection | ✅ |
| 4 | Production hardening; identified 49.7% directional accuracy ceiling | ✅ |
| 5 | Full modularization + trading simulation framework | 🔄 Active |

> Full research notes: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) · Architectural changelog: [docs/project_log.md](docs/project_log.md)

---

## Supplemental Documentation

| Document | Purpose |
|---|---|
| [docs/SYSTEM_MAP.md](docs/SYSTEM_MAP.md) | Living blueprint: every file's inputs, outputs, and dependencies |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Phase 0–4 research history and key breakthroughs |
| [docs/project_log.md](docs/project_log.md) | Automated log of major architectural decisions |
| [.env.template](.env.template) | Environment variable template for Fyers API |
| [.agents/rules/](.agents/rules/) | Enforced coding standards and architectural rules |
