# 🚀 BATCH 1: Quick Start Guide - Data Download

This guide will help you set up the project and download Nifty 50 data.

## ⏱️ Estimated Time: 3-4 hours
- Setup: 30 minutes
- Download 50 stocks: 2.5-3 hours (depends on API speed)
- Verification: 15 minutes

---

## 📋 Prerequisites

### 1. **Fyers Account**
- Active Fyers trading account
- API credentials (Client ID + Secret Key)
- [Get credentials here](https://myapi.fyers.in/dashboard)

### 2. **Python Environment**
- Python 3.10 or higher
- Git installed
- 5GB free disk space (for data)

---

## 🔧 Step-by-Step Setup

### Step 1: Create New Repository

```bash
# Create project directory
mkdir card-stock-prediction
cd card-stock-prediction

# Initialize git
git init

# Create GitHub repo (optional)
gh repo create card-stock-prediction --private --source=. --remote=origin
# OR manually create on GitHub and add remote
git remote add origin https://github.com/YOUR_USERNAME/card-stock-prediction.git
```

### Step 2: Setup Project Structure

```bash
# Copy all files from Batch 1 to project root:
# - README.md
# - requirements.txt
# - .gitignore
# - setup.py
# - config_nifty50.py

# Run setup script
python setup.py
```

**Expected output:**
```
CARD STOCK PREDICTION - PROJECT SETUP
======================================================
Creating directory structure...
  ✅ data/raw
  ✅ data/market_indices
  ✅ data/processed
  ...
  ✅ SETUP COMPLETE
```

### Step 3: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n card-stock python=3.10
conda activate card-stock

# OR using venv
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**This will install:**
- PyTorch 2.1.0 (with CUDA 11.8 if GPU detected)
- Pandas, NumPy, SciPy
- Fyers API
- Technical indicator libraries
- Progress bars, logging, visualization tools

**⚠️ If you get errors:**

```bash
# Try installing PyTorch separately first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install rest
pip install -r requirements.txt
```

### Step 5: Setup Fyers Credentials

```bash
# Create .env file (copy from template)
cp .env.template .env

# Edit .env and add your credentials
nano .env  # or use any text editor
```

**Your .env should look like:**
```
CLIENT_ID=YOUR_CLIENT_ID_HERE
SECRET_KEY=YOUR_SECRET_KEY_HERE
ACCESS_TOKEN=will_be_generated_next
```

### Step 6: Generate Access Token

```bash
# Copy token generation script to scripts/
mkdir -p scripts
cp 00_generate_token.py scripts/

# Run token generator
python scripts/00_generate_token.py
```

**What happens:**
1. Browser opens with Fyers login page
2. Log in and authorize the app
3. Copy `auth_code` from redirected URL
4. Paste into terminal
5. Access token is generated and saved to `.env`

**Example:**
```
FYERS ACCESS TOKEN GENERATION
======================================================
Step 1: Creating Fyers session...
✅ Session created successfully

Step 2: Generating authorization URL...
https://api.fyers.in/api/v2/generate-authcode?...

Step 3: Opening browser for authentication...

Enter auth_code: YOUR_AUTH_CODE_HERE

✅ ACCESS TOKEN GENERATED SUCCESSFULLY
Access Token: eyJ0eXAiOiJKV1QiLCJhbGciOiJIUz...

✅ Access token saved to .env file
```

### Step 7: Verify Setup

```bash
# Check if .env is correct
cat .env

# Should see:
# CLIENT_ID=...
# SECRET_KEY=...
# ACCESS_TOKEN=eyJ0...  (long token)

# Check directory structure
ls -la data/
ls -la scripts/
```

---

## 📥 Download Nifty 50 Data

### Step 8: Copy Download Scripts

```bash
# Copy these files to scripts/ directory:
cp config_nifty50.py .
cp 01_download_nifty50.py scripts/
cp verify_download.py scripts/
```

### Step 9: Review Stock List

```bash
# Check which stocks will be downloaded
python config_nifty50.py
```

**Output:**
```
NIFTY 50 STOCKS
======================================================
Total stocks: 50

First 10:
   1. TCS                  (NSE:TCS-EQ)
   2. INFY                 (NSE:INFY-EQ)
   3. HCLTECH              (NSE:HCLTECH-EQ)
   ...

Index symbol: NSE:NIFTY50-INDEX
```

### Step 10: Start Download

```bash
# Start the download process
python scripts/01_download_nifty50.py
```

**What to expect:**

```
NIFTY 50 DATA DOWNLOAD
======================================================
Date range: 2022-01-01 to 2025-12-31
Chunk size: 30 days
Total chunks per stock: 48
Total stocks to download: 50
======================================================

STEP 1: DOWNLOAD NIFTY 50 INDEX
NIFTY50 INDEX:   0%|          | 0/48 [00:00<?, ?chunk/s]
NIFTY50 INDEX: 100%|██████████| 48/48 [01:23<00:00, 1.74s/chunk]
✅ Nifty 50 index downloaded successfully

STEP 2: DOWNLOAD NIFTY 50 STOCKS

[1/50] Processing TCS...
TCS            : 100%|██████████| 48/48 [01:25<00:00, 1.77s/chunk]
✅ Saved: data/raw/TCS/full.csv
   Total rows: 287,345
   Date range: 2022-01-03 09:15:00 → 2025-12-31 15:30:00

[2/50] Processing INFY...
...
```

**⏱️ Duration:**
- ~1.5-2 minutes per stock (48 chunks × 2 seconds)
- **Total: 2.5-3 hours for all 50 stocks**

**💡 Tip:** Let it run overnight or while doing other work.

### Step 11: Monitor Progress

The script shows:
- ✅ Successful downloads with row counts
- ⚠️ Warnings for missing data
- ❌ Errors if API fails
- 📊 Progress bars for each stock

**Logs saved to:**
```
logs/download_YYYYMMDD_HHMMSS.log
```

### Step 12: Verify Downloaded Data

```bash
# After download completes, verify data quality
python scripts/verify_download.py
```

**Expected output:**
```
DATA VERIFICATION
======================================================
Verifying 50 stocks...
Verifying: 100%|██████████| 50/50 [00:15<00:00,  3.21stock/s]

VERIFICATION SUMMARY
======================================================
Total stocks: 50
Files found: 50/50 ✅
Files missing: 0 ❌
Good quality: 48 ✅
Has issues: 2 ⚠️

Nifty 50 Index: GOOD
  Rows: 289,234
  Range: 2022-01-03 09:15:00 → 2025-12-31 15:30:00

Data Statistics:
  Total rows across all stocks: 14,367,890
  Average rows per stock: 287,358

✅ VERIFICATION COMPLETE
```

---

## 🔍 Understanding the Data

### File Structure

```
data/
├── raw/
│   ├── RELIANCE/
│   │   └── full.csv          # 287k rows, 6 columns
│   ├── TCS/
│   │   └── full.csv
│   └── ... (50 stocks)
└── market_indices/
    └── NIFTY50.csv           # 289k rows, 6 columns
```

### CSV Format

Each `full.csv` contains:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | IST timezone, minute-level |
| open | float | Opening price |
| high | float | Highest price |
| low | float | Lowest price |
| close | float | Closing price |
| volume | int | Volume traded |

**Example rows:**
```
timestamp,open,high,low,close,volume
2022-01-03 09:15:00,2450.0,2455.0,2448.5,2452.3,125430
2022-01-03 09:16:00,2452.3,2453.5,2450.0,2451.0,98234
...
```

### Data Coverage

**Per stock:**
- ~375 rows per trading day (6h 15min of trading)
- ~250 trading days per year
- ~287k rows per stock (4 years)

**Total dataset:**
- 50 stocks × 287k = 14.3M rows
- ~500 MB total size

---

## ✅ Success Criteria - Batch 1

Before moving to Batch 2, verify:

- [x] **50 stock CSV files** in `data/raw/`
- [x] **Nifty 50 index CSV** in `data/market_indices/`
- [x] **All files have >280k rows** (check verification report)
- [x] **No missing data** (or minimal <5% missing)
- [x] **Date range: 2022-01-01 to 2025-12-31**
- [x] **Trading hours only** (9:15 AM - 3:30 PM IST)

---

## 🐛 Troubleshooting

### Issue 1: Access Token Expired

**Error:** `Invalid access token`

**Solution:**
```bash
# Regenerate token
python scripts/00_generate_token.py

# Update .env with new token
```

### Issue 2: API Rate Limit

**Error:** `Rate limit exceeded`

**Solution:**
```bash
# Already handled in script with:
# - 1 second delay between chunks
# - 2 seconds delay between stocks
# - Automatic retries (3 attempts)

# If still failing, increase delays in script:
SLEEP_BETWEEN_CHUNKS = 2.0  # increase from 1.0
SLEEP_BETWEEN_STOCKS = 5.0  # increase from 2.0
```

### Issue 3: Some Stocks Failed

**Error:** `Failed to download STOCK_NAME`

**Solution:**
```bash
# Check logs/download_*.log for specific errors

# Manually retry failed stocks:
# Edit 01_download_nifty50.py to download only failed stocks
# Change NIFTY_50_STOCKS to only include failed ones

# Or wait and retry later (API might be temporarily down)
```

### Issue 4: Nifty Index Not Available

**Error:** `Index file not found`

**Solution:**
```bash
# Try alternative symbol in config_nifty50.py:
NIFTY_INDEX_SYMBOL = "NSE:NIFTY-INDEX"  # instead of NSE:NIFTY50-INDEX

# Or download manually from NSE website
# and place in data/market_indices/NIFTY50.csv
```

### Issue 5: Out of Disk Space

**Error:** `No space left on device`

**Solution:**
```bash
# Check space
df -h

# Each stock ~10MB, need 500MB + 2GB for models later
# Clean up unnecessary files or move to larger drive
```

---

## 📊 What's Next?

After successful Batch 1 completion:

### **Batch 2: Data Processing** (Tomorrow)
- Clean data (remove outliers, gaps)
- Add 18 features (technical indicators + market context)
- Quality checks
- Exploratory analysis

**Files in Batch 2:**
- `02_process_stocks.py` - Feature engineering pipeline
- `exploratory_analysis.ipynb` - Data exploration notebook

**Time estimate:** 4-5 hours

---

## 📞 Questions?

If you encounter any issues:

1. Check `logs/download_*.log` for detailed errors
2. Review `data/verification_report.txt`
3. Verify `.env` file has correct credentials
4. Ensure internet connection is stable

---

## 📝 Summary

**What we accomplished in Batch 1:**
- ✅ Project structure setup
- ✅ Dependencies installed
- ✅ Fyers API authentication
- ✅ Downloaded 50 Nifty stocks (14M+ rows)
- ✅ Downloaded Nifty 50 index
- ✅ Verified data quality

**Ready for Batch 2:** Data Processing & Feature Engineering

---

**Last Updated:** January 30, 2026  
**Batch:** 1 of 4  
**Status:** ✅ COMPLETE
