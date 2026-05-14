# Financial ML Standards (CARD)

## 1. Stationarity
- All input channels MUST be stationary (e.g., log-returns, fractional differences).
- Prices, volumes, or non-stationary indicators are strictly forbidden as raw inputs.

## 2. No Target Leakage
- **RevIN**: Use RevIN (Reversible Instance Normalization) to handle non-stationarity in predictions without leaking future info.
- **Scaling**: Rolling Z-Score scaling MUST be fit ONLY on the training slice and applied to validation/test sets without re-fitting.

## 3. Directional Targets
- The primary target for backtesting and evaluation is **Directional Accuracy** (Up/Down prediction).
- Absolute price prediction is secondary to capturing the sign of the move.
