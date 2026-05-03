# Project Tidy Report
**Date**: 2026-05-02
**Files scanned**: ~65
**Moves executed**: 10
**Violations flagged**: 4 (4 resolved)

## Moves Executed
| From | To | Status |
|------|-----|--------|
| `src/config.py` | `src/utils/config.py` | Executed |
| `src/engine/trainer.py` | `src/training/trainer.py` | Executed |
| `src/engine/lr_scheduler.py` | `src/training/lr_scheduler.py` | Executed |
| `src/engine/__init__.py` | `src/training/__init__.py` | Executed |
| `src/utils/wandb_logger.py` | `archive/wandb_logger.py` | Executed |
| `src/utils/tensorboard_logger.py` | `archive/tensorboard_logger.py` | Executed |
| `demo_ui/` | `archive/demo_ui/` | Executed |
| `src/models/loss.py` | `src/training/loss.py` | Executed |
| `results/` | `archive/results/` | Executed |
| `data/download_statistics.json` | `data/cache/download_statistics.json` | Executed |

## Violations Flagged
| File | Rule | Status |
|------|------|--------|
| `src/models/loss.py` | Model files must be pure architecture | Resolved (Moved to training) |
| `src/features/engine.py` | No monolithic code (>300 lines) | Resolved (Split) |
| `src/data/dataset.py` | No monolithic code (>300 lines) | Resolved (Split) |
| `src/training/trainer.py` | No monolithic code (>300 lines) | Resolved (Split into epoch.py) |

## Ignored Violations & Unknowns
| File | Rule / Issue | Status |
|------|------|--------|
| `data/configs/stock_universe.py` | No monolithic code (>300 lines) | Ignored permanently |
| `.agents/workflows/graphify.md` | .agents/ allows rules/ and skills/ only | Ignored permanently |
| `data/market_indices/` | Unknown directory structure | Ignored for now |

## Import Paths Updated
*Note: The project-tidy skill only flags these. They must be updated manually or by another agent.*
