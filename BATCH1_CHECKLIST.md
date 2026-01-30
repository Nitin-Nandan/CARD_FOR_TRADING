# ✅ BATCH 1: Implementation Checklist

## Phase 1 - Week 1: Data Download

**Goal:** Download and verify Nifty 50 stock data (Jan 2022 - Dec 2025)

---

## 📦 Files Provided

### Core Files
- [ ] `README.md` - Project documentation
- [ ] `requirements.txt` - Python dependencies
- [ ] `.gitignore` - Git ignore rules
- [ ] `setup.py` - Project structure setup
- [ ] `config_nifty50.py` - Stock list configuration

### Scripts
- [ ] `scripts/00_generate_token.py` - Fyers authentication
- [ ] `scripts/01_download_nifty50.py` - Main download script
- [ ] `scripts/verify_download.py` - Data verification

### Documentation
- [ ] `BATCH1_QUICK_START.md` - Step-by-step guide
- [ ] `BATCH1_CHECKLIST.md` - This file

---

## 🔄 Implementation Steps

### Pre-Implementation

- [ ] Create new GitHub repository
- [ ] Clone/setup local project directory
- [ ] Copy all Batch 1 files to project root

### Environment Setup

- [ ] Create virtual environment (conda/venv)
- [ ] Activate environment
- [ ] Run `python setup.py` to create directory structure
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify PyTorch installation with CUDA

### Fyers API Setup

- [ ] Create `.env` file from `.env.template`
- [ ] Add `CLIENT_ID` to `.env`
- [ ] Add `SECRET_KEY` to `.env`
- [ ] Run `python scripts/00_generate_token.py`
- [ ] Verify `ACCESS_TOKEN` saved to `.env`
- [ ] Test API connection (script will confirm)

### Data Download

- [ ] Review stock list: `python config_nifty50.py`
- [ ] Start download: `python scripts/01_download_nifty50.py`
- [ ] Monitor progress (will take 2.5-3 hours)
- [ ] Check for errors in terminal output
- [ ] Review log file in `logs/download_*.log`

### Verification

- [ ] Run verification: `python scripts/verify_download.py`
- [ ] Review console output
- [ ] Check `data/verification_report.txt`
- [ ] Verify all 50 stocks downloaded
- [ ] Verify Nifty 50 index downloaded
- [ ] Confirm data quality (>48 stocks with GOOD status)

### Post-Implementation

- [ ] Commit files to git:
```bash
git add .
git commit -m "Batch 1: Project setup and Nifty 50 data download"
git push origin main
```

- [ ] Document any issues encountered
- [ ] Note download statistics (total rows, time taken)
- [ ] Backup data folder (optional but recommended)

---

## ✅ Success Criteria

### Required (Must Have)

- [x] All 50 Nifty stock CSV files exist
- [x] Nifty 50 index CSV exists
- [x] Each stock file has >280,000 rows
- [x] Date range: 2022-01-01 to 2025-12-31
- [x] No more than 2 stocks with quality issues
- [x] Verification report shows <5% missing data

### Optional (Nice to Have)

- [ ] All stocks with GOOD quality status
- [ ] Zero failed downloads
- [ ] Complete data (no gaps >5 minutes)
- [ ] Log files show no API errors

---

## 📊 Expected Results

### Data Statistics

| Metric | Expected Value |
|--------|---------------|
| **Total Stocks** | 50 |
| **Rows per Stock** | ~287,000 |
| **Total Rows** | ~14.3 million |
| **Disk Space** | ~500 MB |
| **Download Time** | 2.5-3 hours |
| **Date Range** | 2022-01-01 to 2025-12-31 |
| **Trading Days** | ~1,000 days |
| **Rows per Day** | ~375 (6h 15min × 60) |

### File Structure

```
card-stock-prediction/
├── data/
│   ├── raw/
│   │   ├── RELIANCE/full.csv      ✅ ~287k rows
│   │   ├── TCS/full.csv            ✅ ~287k rows
│   │   ├── INFY/full.csv           ✅ ~287k rows
│   │   └── ... (47 more stocks)    ✅
│   └── market_indices/
│       └── NIFTY50.csv             ✅ ~289k rows
├── logs/
│   └── download_*.log              ✅ detailed logs
└── data/verification_report.txt    ✅ quality report
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Import Error

**Symptom:**
```
ModuleNotFoundError: No module named 'fyers_apiv3'
```

**Solution:**
```bash
pip install fyers-apiv3
# or
pip install -r requirements.txt
```

### Issue 2: Access Token Invalid

**Symptom:**
```
❌ API error: Invalid access token
```

**Solution:**
```bash
# Regenerate token (expires after ~24 hours)
python scripts/00_generate_token.py
```

### Issue 3: Download Interrupted

**Symptom:**
```
KeyboardInterrupt or connection error mid-download
```

**Solution:**
```bash
# Script is idempotent - just re-run
# It will skip already downloaded stocks
python scripts/01_download_nifty50.py
```

### Issue 4: Nifty Index Not Found

**Symptom:**
```
⚠️ Nifty 50 Index: NOT FOUND
```

**Solution:**
1. Try alternative symbol in `config_nifty50.py`
2. Or manually download from NSE/Yahoo Finance
3. Save as `data/market_indices/NIFTY50.csv`

---

## 📈 Progress Tracking

### Time Log

| Task | Estimated | Actual | Status |
|------|-----------|--------|--------|
| Environment setup | 30 min | ___ | ⬜ |
| Install dependencies | 15 min | ___ | ⬜ |
| Fyers authentication | 10 min | ___ | ⬜ |
| Download 50 stocks | 180 min | ___ | ⬜ |
| Verify data | 15 min | ___ | ⬜ |
| **Total** | **4h 10min** | ___ | ⬜ |

### Completion Tracking

| Stock | Downloaded | Verified | Rows | Issues |
|-------|-----------|----------|------|--------|
| TCS | ⬜ | ⬜ | ___ | ___ |
| INFY | ⬜ | ⬜ | ___ | ___ |
| RELIANCE | ⬜ | ⬜ | ___ | ___ |
| ... | ... | ... | ... | ... |

(Use verification report to fill this)

---

## 🎯 Ready for Batch 2?

Before proceeding to Batch 2 (Data Processing), ensure:

### Critical
- [x] All 50 stocks downloaded successfully
- [x] Nifty index downloaded
- [x] Verification passed (≤2 stocks with issues)
- [x] Data backed up (optional but recommended)

### Recommended
- [x] Review verification report details
- [x] Check sample data looks correct (open CSV)
- [x] Understand data structure (timestamp, OHLCV)
- [x] Git commit and push complete

### Optional
- [ ] Explore data in Jupyter notebook
- [ ] Calculate basic statistics (mean, std)
- [ ] Plot sample stock price chart
- [ ] Check for distribution shifts across years

---

## 📝 Notes & Observations

**Space for your notes:**

**Download statistics:**
- Start time: ___________
- End time: ___________
- Total duration: ___________
- Failed stocks: ___________
- API errors encountered: ___________

**Data quality observations:**
- Stocks with issues: ___________
- Reason for issues: ___________
- Actions taken: ___________

**Next steps:**
- Issues to resolve before Batch 2: ___________
- Questions for mentor: ___________

---

## 🚀 Next Batch Preview

### Batch 2: Data Processing (Tomorrow)

**What we'll build:**
1. **Processing pipeline** - Clean data, remove outliers
2. **Feature engineering** - Add 18 features:
   - EMA, ROI, Bollinger Bands
   - Nifty correlation
   - Time of day encoding
   - Volatility metrics
3. **Quality checks** - Automated validation
4. **Exploratory analysis** - Jupyter notebook

**Files to create:**
- `scripts/02_process_stocks.py`
- `notebooks/exploratory_analysis.ipynb`
- Processing configuration file

**Estimated time:** 4-5 hours

---

**Batch:** 1 of 4  
**Status:** 🔄 IN PROGRESS  
**Updated:** January 30, 2026
