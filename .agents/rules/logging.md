# Logging & Terminal Hygiene

## 1. Clean Terminal Protocol
- **Console Output**: Only WARNING level and above, plus high-level progress indicators (e.g., "Epoch 1/50", "Step 100/500").
- **Diagnostic Silence**: Never print raw metrics, shapes, or debug info to the console.
- **Progress-Only**: The terminal should look like a professional dashboard, not a raw log dump.

## 2. Dual-Channel Logging
- **`RunLogger`**: Use the project's `RunLogger` (or standard `logging` configured for it).
- **Files**: All DEBUG and INFO details MUST go to `logs/<run_id>/<run_id>.log`.
- **Metrics**: Use `logger.metric()` to track values for file logging without cluttering the screen.

## 3. Forbidden Output
- No `print()` statements in production code. Use `logger.info()` or `logger.warning()`.
- No `loguru` (deprecated for this project). Use standard library `logging`.
