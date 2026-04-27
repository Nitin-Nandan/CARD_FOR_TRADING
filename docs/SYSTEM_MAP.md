# CARD System Map

> Living architectural blueprint. Last updated: 2026-03-23.

---

## `src/` — Core Library

| Filepath | Core Purpose | Inputs | Outputs | Dependencies |
|---|---|---|---|---|
| `src/config.py` | Central hyperparameter & path configuration | None (hardcoded defaults) | `Config` dataclass | `pipeline/*`, `scripts/training/*`, `src/engine/trainer.py` |
| `src/data/downloader.py` | Downloads historical OHLCV data via Fyers API | Fyers API, `.env` credentials | `.csv` files → `data/raw/` | `pipeline/01_download.py` |
| `src/data/builder.py` | Creates sliding windows from processed stock data | `.parquet` from `data/processed/` | `_windows.npz` → `data/windows/` | `pipeline/03_windowing.py` |
| `src/data/dataset.py` | PyTorch `Dataset` for training — loads windows via mmap | `.npy`/`.npz` from `data/windows/` | `dict{X, y, y_close, last_price, stock}` tensors | `src/engine/trainer.py` |
| `src/features/engine.py` | Feature engineering (EMA, BB, Nifty, volume, time) | `.csv` from `data/raw/`, Nifty index | `.parquet` → `data/processed/` | `pipeline/02_process.py` |
| `src/models/card.py` | Top-level CARD model assembly (encoder + heads) | `(B, 81, 60)` input tensor | `(B, 15)` close predictions, `(B, 15)` return predictions | `src/engine/trainer.py`, `scripts/serving/demo_server.py` |
| `src/models/layers.py` | RevIN normalization + ConvNeXtV2 blocks | Tensors from `card.py` | Normalized/transformed tensors | `src/models/card.py` |
| `src/models/attention.py` | Multi-Scale Temporal Attention (MSTA) module | Feature tensors | Attention-weighted tensors | `src/models/card.py` |
| `src/models/loss.py` | `CombinedReturnLoss` (direction + magnitude) | Predicted & actual returns | Scalar loss value | `src/engine/trainer.py` |
| `src/engine/trainer.py` | Full training loop (train/val, checkpointing, logging) | `Config`, `Dataset`, `CARD` model | Checkpoints → `checkpoints/`, logs, metrics | `scripts/training/production_6d.py` |
| `src/engine/lr_scheduler.py` | Cosine annealing + linear warmup scheduler | Optimizer, epoch count | Updated learning rates | `src/engine/trainer.py` |
| `src/utils/auth.py` | Fyers API authentication & token management | `.env` credentials, browser auth flow | Updated `ACCESS_TOKEN` in `.env` | `pipeline/00_auth.py` |
| `src/utils/logger.py` | Multi-backend logger (file + TensorBoard + W&B) | Metric dicts, config | Log entries to `logs/`, TensorBoard, W&B | `src/engine/trainer.py` |
| `src/utils/tensorboard_logger.py` | TensorBoard scalar logging wrapper | Metric name/value pairs | TensorBoard event files | `src/utils/logger.py` |
| `src/utils/wandb_logger.py` | Weights & Biases logging wrapper (offline mode) | Config, metrics | W&B run data → `wandb/` | `src/utils/logger.py` |

---

## `pipeline/` — Data Factory

| Filepath | Core Purpose | Inputs | Outputs | Dependencies |
|---|---|---|---|---|
| `pipeline/00_auth.py` | Step 0: Generate Fyers API access token | User browser auth code | `ACCESS_TOKEN` saved to `.env` | Uses `src/utils/auth.py` |
| `pipeline/01_download.py` | Step 1: Download historical 1-min OHLCV data | Fyers API, `data/configs/stock_universe.py` | `.csv` files → `data/raw/{STOCK}/full.csv` | Uses `src/data/downloader.py` |
| `pipeline/02_process.py` | Step 2: Feature engineering on raw data | `.csv` from `data/raw/`, Nifty index data | `.parquet` → `data/processed/{STOCK}_processed.parquet` | Uses `src/features/engine.py` |
| `pipeline/03_windowing.py` | Step 3: Create sliding windows for training | `.parquet` from `data/processed/` | `_windows.npz` → `data/windows/` | Uses `src/data/builder.py` |

---

## `scripts/` — Triggers & Tools

### `scripts/training/`

| Filepath | Core Purpose | Inputs | Outputs | Dependencies |
|---|---|---|---|---|
| `scripts/training/production_6d.py` | 6-day production training entrypoint | `src/config.py`, `data/windows/*.npz` | Checkpoints, logs, W&B runs | Uses `src/config.py`, `src/engine/trainer.py` |

### `scripts/serving/`

| Filepath | Core Purpose | Inputs | Outputs | Dependencies |
|---|---|---|---|---|
| `scripts/serving/demo_server.py` | Flask prediction server with live Fyers data | Fyers API, `checkpoints/*.pt` | JSON predictions via REST API | Uses `src/models/card.py`, `train.config` |

### `scripts/data_ops/`

| Filepath | Core Purpose | Inputs | Outputs | Dependencies |
|---|---|---|---|---|
| `scripts/data_ops/audit_data.py` | Pre-training data sanity check (NaN, raw prices) | `data/windows/*.npz` | Console report | Standalone |
| `scripts/data_ops/clean_all_data.py` | Wipes all pipeline-generated data | `data/raw/`, `data/processed/`, `data/windows/` | Empty directories | Standalone |
| `scripts/data_ops/analyze_history.py` | Analyzes training history JSON logs | `logs/*.json` | Console summary | Standalone |
| `scripts/data_ops/verify_download.py` | Validates downloaded raw CSV quality | `data/raw/*.csv` | Console + `logs/reports/verification_report.txt` | Uses `data/configs/stock_universe.py` |
| `scripts/data_ops/verify_processed.py` | Validates processed parquet quality | `data/processed/*.parquet` | Console + `logs/reports/processed_verification_report.txt` | Uses `data/configs/stock_universe.py` |

### `scripts/validation/`

| Filepath | Core Purpose | Inputs | Outputs | Dependencies |
|---|---|---|---|---|
| `scripts/validation/hyperparam_sweep.py` | Hyperparameter search across configs | `src/config.py`, training data | Sweep results, best config | Uses `src/engine/trainer.py` |
| `scripts/validation/validate_features.py` | Validates feature distributions after processing | `data/processed/*.parquet` | Console report | Standalone |
| `scripts/validation/validate_windows.py` | Validates window shapes and statistical properties | `data/windows/*.npz` | Console report | Standalone |

### `scripts/diagnostics/`

| Filepath | Core Purpose | Inputs | Outputs | Dependencies |
|---|---|---|---|---|
| `scripts/diagnostics/audit_config.py` | Analyzes NSE 500 stock config quality | `data/configs/stock_universe.py` | Console report (dupes, dates, sectors) | Uses `data/configs/stock_universe.py` |
| `scripts/diagnostics/debug_grads.py` | Gradient flow debugger | Model, sample batch | Console gradient stats | Uses `src/models/` |
| `scripts/diagnostics/debug_ram.py` | RAM usage profiler for data loading | `data/windows/*.npz` | Console memory report | Standalone |
| `scripts/diagnostics/run_diagnostics.py` | Master diagnostics runner | Various | Aggregated diagnostic output | Uses other diagnostic scripts |
