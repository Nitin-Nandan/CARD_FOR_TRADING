# Project CARD: Development History (Phases 0-4)

[Modular Architect Active]

This document serves as the consolidated single source of truth for the evolution of Project CARD, from raw data ingestion to production-grade modular architecture and high-frequency predictive modeling.

---

## 1. Data Integrity & Storage Optimization (Phase 0)
**The Challenge:** Initial attempts to generate sliding windows for 504 stocks over multiple years threatened to consume **1.3 Terabytes** of disk space and crash RAM during training.

**The Breakthrough: Index-Based Windowing & Gap Detection**
- **Gap Detection:** Implemented vectorized `time_diff > 2 min` checks to identify overnight gaps, weekends, and holidays.
- **Sliding Window View:** Utilized NumPy's `sliding_window_view` to locate contiguous 75-minute blocks (60m input + 15m target) without internal time gaps or NaNs.
- **Storage Solution:** Switched from saving physical overlapped arrays to an **Index-Based Strategy**. We save the raw 1D data array once and a list of `valid_indices`.
- **Result:** Storage plummeted from 1.3 TB to **~2 GB**, with zero compromise on data granularity.

---

## 2. Advanced Feature Engineering (Phases 1 & 3)
**The Objective:** Unleash the "Channel-Aligned" power of the CARD Transformer by moving from 5 raw OHLCV channels to a rich 81-channel feature space.

**The 81-Channel Engine:**
- **Momentum (15):** RSI, MACD, Stochastics, ROC, Williams %R.
- **Volatility (10):** ATR, True Range, Parkinson/Garman-Klass Volatility, 30-day Realized Vol.
- **Volume (12):** SMAs, OBV, Money Flow Index (MFI), Force Index, VWAP Distances.
- **Cross-Asset (5):** Real-time Relative Strength and Running Betas (`beta_60`, `beta_20`) against the NIFTY 50 Index.
- **Time/Pattern (10):** `is_monday`, `minutes_of_day` (sin/cos), 20-day trailing mathematical High/Low proximities.

**Phase 3: Regime Detection Features**
- **Volatility Regime:** Current 30m Vol / 5-day Median Vol (Identifies calm vs. chaotic states).
- **Trend Regime:** `(SMA50 / SMA200) - 1` (Identifies macro bull/bear states).
- **Time Regime:** Boolean flag for the high-volatility Open (9:15-10:15) and Close (14:30-15:30) sessions.

---

## 3. Model Evolution: The Pivot to Returns (Phase 2)
**The Objective:** Replace raw price prediction (non-stationary) with financial returns (stationary) to maximize mathematical stability.

**Logic Shift:**
- **Stationarity:** Predicting `(future_prices / current_price) - 1`. This allows the model to learn percentages rather than absolute Rupee values, which vary wildly across the Nifty 500.
- **CombinedReturnLoss:** A specialized dual-objective loss function:
    1. **Signal Decay MAE:** Magnitude loss scaled by 1000x to maintain gradient signal, with decay weighting to prioritize near-term (5-10 min) accuracy.
    2. **Directional Penalty:** Explicit penalty for sign mismatch (+/-), forcing the model to prioritize getting the "direction" right.
- **Safety Clamping:** Implemented a ±20% return clamp (matching Indian market circuit breakers) in the forward pass to prevent gradient explosions from extreme/erroneous predictions.

---

## 4. Production Optimizations & Results (Phase 4)
**The Results: The "49.7% Ceiling"**
- **Infrastructure:** Success with **SSD mmap streaming** and **Metadata Slicing** allowed 0-MB RAM overhead for dataset loading.
- **Performance:** Throughput reached ~4.0 iterations/sec on standard hardware with high-liquidity stocks (TCS, RELIANCE).
- **The Ceiling:** Extensive hyperparameter sweeps on TCS revealed a directional accuracy (DA) plateau at **49.7%**.

**Post-Mortem Findings:**
- **Market Efficiency:** At 1-minute resolution, noise dominates the signal. Most professional HFT algorithms operate in the 51-53% range; 49.7% represents a "near-random" baseline that requires **Meta-Labeling** or **Regime-Switching** to break into profitability.
- **The Pivot:** The project transitioned from seeking raw binarized accuracy to measuring **Alpha (ROI)** on high-confidence breakout moves.

---

## 5. Final Modular Architecture
As of March 20, 2026, the project is finalized into a production-ready library:
- `src/`: Core truth (Models, Engine, Features, Utils).
- `pipeline/`: Standardized wrappers for Download -> Process -> Window.
- `scripts/`: Diagnostic and Training triggers.
- `docs/`: Automated logs and this Development History.
