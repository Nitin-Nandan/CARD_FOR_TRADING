# Archive: Intraday V1

**Archived on**: 2026-05-03  
**Reason**: Project pivoted from 15-min intraday (Fyers API) to daily NSE stock prediction (yfinance).  
**Status**: Superseded — do not import. Preserved for reference only.

## What was here

| File | Description |
|------|-------------|
| `pipeline/00_auth.py` | Fyers OAuth authentication flow |
| `pipeline/01_download.py` | Fyers 15-min OHLCV downloader |
| `pipeline/02_process.py` | 81-channel intraday feature engine runner |
| `pipeline/03_windowing.py` | .npz window builder (seq_len=60, pred_len=15) |
| `src/data/downloader.py` | FyersDownloader class |
| `src/data/normalization.py` | Numpy clip normalization (minimal) |
| `src/data/dataset.py` | Intraday .npz window dataset |
| `src/data/builder.py` | WindowBuilder for .npz files |
| `src/data/samplers.py` | Balanced sampling for intraday windows |
| `src/features/engine.py` | 81-channel FeatureEngine (intraday regime/tick) |
| `src/features/momentum.py` | Intraday momentum features |
| `src/features/volatility.py` | Intraday volatility features |
| `src/features/volume.py` | Intraday volume features |
| `src/utils/config.py` | Old Config: enc_in=81, seq_len=60, pred_len=15, 15-min |

## Superseded by

- `data/configs/daily_universe.py` — 25 NSE stocks for yfinance
- `pipeline/10_fetch_daily.py` — yfinance daily downloader
- `src/data/transforms.py` — LogReturnTransform
- `src/features/daily_features.py` — 15 stationary daily features
- `src/data/daily_dataset.py` — daily sliding-window dataset
- `src/utils/daily_config.py` — DailyConfig: enc_in=15, seq_len=120, pred_len=5
- `pipeline/11_train_daily.py` — end-to-end training pipeline
