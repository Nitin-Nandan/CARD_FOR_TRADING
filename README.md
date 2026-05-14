# CARD: Channel Aligned Robust Blend Transformer for NSE

> [!IMPORTANT]
> This project has transitioned from **15-minute intraday** (Fyers API) to **Daily NSE stock prediction** (yfinance). The legacy intraday pipeline is preserved in `archive/intraday_v1/`.

A production-grade quantitative ML pipeline for daily return prediction on the NSE, based on the [ICLR 2024 CARD paper](https://openreview.net/forum?id=MJksrOhurza). CARD addresses key shortcomings of channel-independent Transformers by introducing channel-aligned attention, a multi-scale token blend module, and a robust uncertainty-weighted loss.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📈 What This Project Does

CARD predicts **5-day forward log-returns** for a universe of 25 NSE stocks using stationary daily features and a robust Transformer architecture.

- **Data Acquisition** — yfinance → 25 NSE Liquid Stocks (Daily OHLCV)
- **Stationarity** — All inputs transformed to log-returns or stationary indicators to prevent spurious correlations.
- **Architecture** — Channel-Aligned Robust Blend (CARD):
  - **Channel-Aligned Attention**: Captures temporal correlations + dynamical dependence among variables.
  - **Token Blend Module**: Efficiently utilizes multi-scale knowledge without signal decomposition.
  - **Robust Loss**: Uncertainty-weighted loss that prioritizes near-horizon accuracy.
- **Governance** — Strict architectural and hygiene rules enforced by AI agents in `.agents/rules/`.

---

## 🏗️ Project Structure

The project follows a **Library-Factory-Trigger** pattern:

```text
CARD_FOR_TRADING/
├── src/                    # Core library
│   ├── data/               # DailyDataset, LogReturnTransform
│   ├── features/           # 15-channel daily stationary features
│   ├── training/           # Epoch engine, Evaluator, Loss functions
│   └── utils/              # RunLogger, DailyConfig
├── pipeline/               # Execution factory
│   ├── fetch_daily.py      # yfinance downloader
│   └── train_daily.py      # End-to-end training pipeline
├── data/
│   └── configs/            # Stock universe (25 NSE stocks)
├── docs/                   
│   ├── reference/          # ICLR 2024 CARD Paper (card_paper.txt)
│   └── research/           # Implementation notes and breakthroughs
└── archive/                
    └── intraday_v1/        # Legacy 15-min Fyers-based pipeline
```

> [!TIP]
> See [docs/WHAT_WE_KNOW.md](docs/WHAT_WE_KNOW.md) for the latest project status and [docs/EXPERIMENT_REGISTRY.md](docs/EXPERIMENT_REGISTRY.md) for training results.

---

## 🚀 Quick Start

### 1. Set Up Environment
```bash
# Create and activate environment
conda create -n card python=3.10 -y
conda activate card

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Daily Pipeline
The pipeline consists of two primary triggers:

```bash
# Step 1: Fetch Daily OHLCV data for the universe
python pipeline/fetch_daily.py

# Step 2: Train the CARD model
python pipeline/train_daily.py
```

---

## 🏛️ Legacy Archive (Intraday V1)

We have transitioned away from the high-frequency intraday pipeline to focus on daily stationarity. The old infrastructure is archived in `archive/intraday_v1/`.

| Feature | Intraday V1 (Legacy) | Daily V2 (Current) |
| :--- | :--- | :--- |
| **Resolution** | 15-Minute | Daily |
| **Data Source** | Fyers API (OAuth) | yfinance |
| **Channels** | 81 (Mixed stationarity) | 15 (Strictly stationary) |
| **Horizon** | 15 steps (3.75 hours) | 5 steps (1 week) |
| **Lookback** | 60 steps | 120 steps |

> [!NOTE]
> The legacy version achieved a ~49.7% directional accuracy ceiling but struggled with non-stationary drift in the 81-feature engine.

---

## 📚 Supplemental Documentation

| Document | Purpose |
| :--- | :--- |
| [docs/EXPERIMENT_REGISTRY.md](docs/EXPERIMENT_REGISTRY.md) | Tracking training runs, hyperparams, and metrics. |
| [docs/WHAT_WE_KNOW.md](docs/WHAT_WE_KNOW.md) | Living status document and architectural audit log. |
| [docs/reference/card_paper.txt](docs/reference/card_paper.txt) | Local text version of the ICLR 2024 CARD paper. |
| [.agents/rules/](.agents/rules/) | Governance layer: `financial_ml.md`, `governance.md`, `hygiene.md`. |
