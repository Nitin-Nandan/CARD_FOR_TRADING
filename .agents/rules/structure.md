# Project Structure & Naming

## 1. Directory Hierarchy
Every file MUST belong to one of these canonical directories:
- `data/`: 
  - `raw/`: Untouched downloads (READ ONLY).
  - `processed/`: Feature-engineered datasets.
  - `cache/`: Intermediate artifacts (mmap, etc.).
  - `configs/`: Universe definitions, date ranges, splits.
- `src/`:
  - `data/`: Ingestion, parsing, validation.
  - `features/`: Engineering logic ONLY.
  - `models/`: Pure architecture (ZERO training/loss logic).
  - `training/`: Loops, schedulers, loss functions.
  - `evaluation/`: Metrics, backtesting.
  - `utils/`: Loggers, config loaders.
- `pipeline/`: Orchestration scripts ONLY.
- `scripts/`: One-off helpers.
- `archive/`: Legacy code (not for active use).
- `docs/`: Documentation.

## 2. Naming Conventions
- **Orchestrators**: Use intuitive names (e.g., `train_daily.py`) instead of numbered prefixes (e.g., `11_...`).
- **Scripts**: Must be descriptive. Avoid `utils.py` or `misc.py` at the root.
- **Consistency**: Files serving similar purposes across stocks/scales must share naming patterns.
