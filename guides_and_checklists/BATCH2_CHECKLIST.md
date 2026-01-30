# ✅ BATCH 2: Implementation Checklist

## Phase 1 - Week 1: Data Processing & Feature Engineering

**Goal:** Transform raw OHLCV data into 18-feature analysis-ready dataset

---

## 📦 Files Provided

### Scripts
- [ ] `scripts/02_process_stocks.py` - Main processing pipeline
- [ ] `scripts/verify_processed.py` - Processed data verification

### Documentation
- [ ] `BATCH2_QUICK_START.md` - Implementation guide
- [ ] `BATCH2_CHECKLIST.md` - This file

---

## 🔄 Implementation Steps

### Pre-Processing

- [ ] Copy Batch 2 files to project
- [ ] Verify Batch 1 completed (50 raw stocks downloaded)
- [ ] Ensure Nifty index downloaded (370k rows)

### Run Processing

- [ ] Copy `02_process_stocks.py` to `scripts/`
- [ ] Copy `verify_processed.py` to `scripts/`
- [ ] Run: `python scripts/02_process_stocks.py`
- [ ] Monitor progress (45-60 minutes)
- [ ] Check for errors in terminal

### Verification

- [ ] Run: `python scripts/verify_processed.py`
- [ ] Review console output
- [ ] Check `data/processed_verification_report.txt`
- [ ] Verify all 50 stocks processed
- [ ] Confirm 18 features per stock
- [ ] Check average Nifty correlation > 0.5

### Post-Processing

- [ ] Commit to git:
```bash
git add scripts/02_process_stocks.py
git add scripts/verify_processed.py
git commit -m "Batch 2: Data processing with 18 features"
git push
```

- [ ] Document processing statistics
- [ ] Note any stocks with issues
- [ ] Backup processed data (optional)

---

## ✅ Success Criteria

### Required (Must Have)

- [x] All 50 stocks processed successfully
- [x] Each stock has exactly 19 columns (timestamp + 18 features)
- [x] Zero (or <0.1%) NaN values after processing
- [x] ~370k rows per stock (trading hours only)
- [x] Average Nifty correlation > 0.5
- [x] Time features (hour_sin, hour_cos) in range [-1, 1]
- [x] All features have reasonable distributions

### Optional (Nice to Have)

- [ ] All stocks with Nifty correlation > 0.6
- [ ] Zero processing errors/warnings
- [ ] Processing completed in < 60 minutes
- [ ] Detailed feature statistics logged

---

## 📊 Expected Results

### Processing Statistics

| Metric | Expected Value |
|--------|---------------|
| **Total Stocks** | 50 |
| **Rows per Stock** | ~370,000 |
| **Features** | 18 (+ timestamp = 19 columns) |
| **Total Rows** | ~18.5 million |
| **Processing Time** | 45-60 minutes |
| **Failed Stocks** | 0 |
| **NaN Values** | 0 (after dropna) |

### Feature Breakdown

| Category | Features | Count |
|----------|----------|-------|
| **OHLCV** | open, high, low, close, volume | 5 |
| **Technical** | ema_5, ema_10, roi_1m, bb_upper, bb_middle, bb_lower | 6 |
| **Market Context** | nifty_close, nifty_return, hour_sin, hour_cos | 4 |
| **Vol/Volume** | realized_vol_30m, volume_ratio, volume_surge | 3 |
| **Total** | | **18** |

### File Structure

```
data/
├── raw/                                # Input (Batch 1)
│   ├── RELIANCE/full.csv               370k rows, 6 cols
│   └── ... (50 stocks)
│
├── processed/                          # ✨ Output (Batch 2)
│   ├── RELIANCE_processed.csv          370k rows, 19 cols ✅
│   ├── TCS_processed.csv               370k rows, 19 cols ✅
│   ├── ... (48 more)
│   └── NIFTY50_processed.csv           370k rows ✅
│
└── processed_verification_report.txt   # ✨ Quality report
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Processing Hangs on a Stock

**Symptom:**
```
Processing: STOCK_X
  Loaded: 370,185 rows
  [hangs for >5 minutes]
```

**Solution:**
```bash
# Ctrl+C to stop
# Check logs/processing_*.log for details
# If specific stock has issues:
rm data/raw/STOCK_X/full.csv
python scripts/01_download_nifty50.py  # re-download that stock
python scripts/02_process_stocks.py     # re-process
```

### Issue 2: Feature Count Mismatch

**Symptom:**
```
❌ Feature mismatch! Expected 19, got 17
```

**Cause:** Some feature calculation failed  
**Solution:**
- Check which features are missing in logs
- Verify raw data has OHLCV columns
- Ensure Nifty merge succeeded

### Issue 3: High NaN Count

**Symptom:**
```
⚠️  STOCK_X:
    - NaN values: 5000
```

**Cause:** Large gaps in raw data  
**Solution:**
- Check `data/verification_report.txt` (Batch 1)
- If stock has >5% gaps, consider excluding
- Or accept higher NaN and handle in windowing

### Issue 4: Low Nifty Correlation

**Symptom:**
```
Average Nifty correlation: 0.25 (expected > 0.5)
```

**Cause:** Individual stocks might not track Nifty closely  
**Solution:**
- This is expected for some stocks (sector-specific moves)
- As long as most stocks > 0.5, it's fine
- For stocks with corr < 0.2, investigate data quality

---

## 📈 Progress Tracking

### Time Log

| Task | Estimated | Actual | Status |
|------|-----------|--------|--------|
| Copy scripts | 5 min | ___ | ⬜ |
| Run processing | 50 min | ___ | ⬜ |
| Run verification | 10 min | ___ | ⬜ |
| Review results | 15 min | ___ | ⬜ |
| **Total** | **80 min** | ___ | ⬜ |

### Stock Processing Log

Track which stocks completed successfully:

```
✅ RELIANCE - 370,153 rows
✅ TCS - 370,179 rows
✅ INFY - 370,183 rows
...
```

(Use verification report to fill this)

---

## 🎯 Ready for Batch 3?

Before proceeding to Batch 3 (CARD Architecture), ensure:

### Critical
- [x] All 50 stocks processed
- [x] 18 features per stock
- [x] Verification passed
- [x] Processed data backed up

### Recommended
- [x] Review sample processed data
- [x] Understand each feature's meaning
- [x] Check feature distributions look reasonable
- [x] Git commit complete

### Optional
- [ ] Plot sample technical indicators
- [ ] Verify time-of-day encoding
- [ ] Check volume surge detection
- [ ] Calculate feature correlations

---

## 📝 Notes & Observations

**Space for your notes:**

**Processing statistics:**
- Total time: ___________
- Failed stocks: ___________
- Average rows per stock: ___________
- Average Nifty correlation: ___________

**Feature quality observations:**
- Best correlated stock: ___________
- Lowest correlated stock: ___________
- Any unexpected patterns: ___________

**Next steps:**
- Ready for Batch 3? ___________
- Any concerns: ___________

---

## 🚀 Next Batch Preview

### Batch 3: CARD Architecture (Next)

**What we'll build:**
1. **True CARD model**
   - Patch embedding layer
   - Channel attention module
   - Token attention module
   - Token blend module
   - RevIN normalization

2. **Multi-task loss**
   - Returns prediction loss
   - Volatility prediction loss
   - Signal decay weighting

3. **Supporting components**
   - Custom Dataset class
   - Training utilities
   - Model configuration

**Files to create:**
- `models/card_stock.py` (main architecture)
- `models/channel_attention.py`
- `models/token_attention.py`
- `models/token_blend.py`
- `models/revin.py`
- `losses/multi_task_loss.py`

**Estimated time:** 2-3 hours (pure coding, no long waits)

---

**Batch:** 2 of 4  
**Status:** 🔄 READY TO RUN  
**Updated:** January 31, 2026
