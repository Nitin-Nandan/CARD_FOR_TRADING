# ✅ BATCH 3: Implementation Checklist

## Phase 1 - Week 2: CARD Architecture + Windowing

**Goal:** Build complete CARD model and create 60-min training windows

---

## 📦 Files Provided

### Model Architecture
- [ ] `card_complete.py` - All CARD components in one file
  - RevIN (normalization)
  - Channel Attention
  - Token Attention
  - Token Blend
  - CARDStock (main model)
  - MultiTaskLoss

### Pipeline Scripts
- [ ] `03_create_windows_60min.py` - Windowing script (stride=1)

### Documentation
- [ ] `BATCH3_QUICK_START.md` - Implementation guide
- [ ] `BATCH3_CHECKLIST.md` - This file

---

## 🔄 Implementation Steps

### Part 1: Model Setup

- [ ] Create `models/` directory
- [ ] Create `losses/` directory
- [ ] Create `__init__.py` in both directories
- [ ] Copy `card_complete.py` to `models/`
- [ ] Test model:
```bash
cd models
python card_complete.py
```

**Expected output:**
```
Testing CARD Stock Model
Model parameters: 2,547,341
Input: torch.Size([4, 60, 18])
Returns: torch.Size([4, 15])
Volatility: torch.Size([4, 15])
✅ All components working!
```

### Part 2: Windowing

- [ ] Copy `03_create_windows_60min.py` to `pipeline/`
- [ ] Verify Batch 2 completed (50 processed stocks exist)
- [ ] Run windowing:
```bash
python pipeline/03_create_windows_60min.py
```

- [ ] Monitor progress (30-60 minutes)
- [ ] Check for errors in logs

### Part 3: Verification

- [ ] Verify all 50 stocks windowed
- [ ] Check `data/windows/` directory created
- [ ] Verify .npz files (~250-300 MB each)
- [ ] Quick inspection:
```python
import numpy as np
data = np.load('data/windows/RELIANCE_windows.npz')
print(f"Windows: {len(data['X']):,}")
print(f"X shape: {data['X'].shape}")
print(f"y_returns shape: {data['y_returns'].shape}")
```

### Part 4: Documentation

- [ ] Move guides to `guides_and_checklists/`:
```bash
mv BATCH3_QUICK_START.md guides_and_checklists/
mv BATCH3_CHECKLIST.md guides_and_checklists/
```

- [ ] Git commit:
```bash
git add models/ pipeline/03_create_windows_60min.py
git commit -m "Batch 3: CARD architecture and windowing"
git push
```

---

## ✅ Success Criteria

### Required (Must Have)

#### Model Architecture
- [x] Model file exists and imports successfully
- [x] Model has ~2.5M parameters
- [x] Forward pass works: (batch, 60, 18) → (batch, 15) for both returns and volatility
- [x] Loss computation works without errors
- [x] Test script passes all checks

#### Windowing
- [x] All 50 stocks successfully windowed
- [x] Total ~18.5M windows created
- [x] Each stock has ~369k windows
- [x] Files saved in `data/windows/` directory
- [x] X shape: (N, 60, 18) ✅
- [x] y_returns shape: (N, 15) ✅
- [x] y_volatility shape: (N, 15) ✅

#### Data Quality
- [x] No NaN values in any window
- [x] Returns in reasonable range (-0.1 to 0.1)
- [x] Volatility all positive values
- [x] Timestamps sequential and valid

### Optional (Nice to Have)

- [ ] Model components split into separate files (cleaner)
- [ ] CUDA compatibility tested (if GPU available)
- [ ] Sample window visualizations created
- [ ] Window statistics computed and logged

---

## 📊 Expected Results

### Model Specifications

| Component | Details |
|-----------|---------|
| **Total Parameters** | ~2,547,341 |
| **Input Shape** | (batch, 60, 18) |
| **Output Shapes** | Returns: (batch, 15)<br>Volatility: (batch, 15) |
| **Memory (batch=64)** | ~4.5 GB VRAM |
| **Patch Config** | 8-min patches, stride=4 → 13 tokens |
| **Attention Layers** | 2× Channel + 2× Token |

### Windowing Statistics

| Metric | Expected Value |
|--------|---------------|
| **Total Stocks** | 50 |
| **Windows per Stock** | ~369,000 |
| **Total Windows** | ~18,500,000 |
| **Input Window Size** | 60 minutes |
| **Prediction Horizon** | 15 minutes |
| **Stride** | 1 minute (dense sampling) |
| **Features** | 18 |
| **File Size (per stock)** | ~250-300 MB |
| **Total Data Size** | ~13.5 GB |

### File Structure

```
data/windows/
├── RELIANCE_windows.npz          280 MB
│   ├── X: (369,017, 60, 18)
│   ├── y_returns: (369,017, 15)
│   ├── y_volatility: (369,017, 15)
│   ├── timestamps: (369,017,)
│   └── feature_names: ['open', 'high', ...]
│
├── TCS_windows.npz                275 MB
├── INFY_windows.npz               278 MB
└── ... (47 more stocks)

Total: ~13.5 GB
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Model Test Fails with Shape Mismatch

**Symptom:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**Cause:** Input shape incorrect or model config mismatch

**Solution:**
```python
# Check input is exactly (batch, 60, 18)
x = torch.randn(4, 60, 18)

# Verify model config
model = CARDStock(
    num_features=18,  # Must match!
    seq_len=60,       # Must match!
    pred_len=15,
    d_model=128,
    n_heads=8
)
```

### Issue 2: Windowing Runs Out of Memory

**Symptom:**
```
MemoryError: Unable to allocate 2.1 GiB
```

**Cause:** Creating too many windows at once

**Solution:**
- Close other programs
- Windowing uses ~4GB RAM temporarily
- System needs 16GB total RAM
- If still fails, process stocks in smaller batches

### Issue 3: Windowing Too Slow

**Symptom:**
```
Processing RELIANCE: [stuck at 15% for 10 minutes]
```

**Expected behavior:**
- ~40-60 seconds per stock is normal
- 50 stocks × 50 seconds = ~42 minutes total
- NOT a problem! This is dense sampling (18.5M windows)

**To speed up (optional):**
- Use stride=15 instead of stride=1 (reduces to 1.2M windows)
- Trade-off: Less training data

### Issue 4: Window NaN Values

**Symptom:**
```
Found NaN in window X[1234]
```

**Cause:** Source processed data had gaps

**Solution:**
1. Check source file: `data/processed/STOCK_processed.csv`
2. Re-run processing (Batch 2) for that stock
3. Or exclude the stock from training

### Issue 5: CUDA Out of Memory (Model Testing)

**Symptom:**
```
CUDA out of memory. Tried to allocate 512.00 MiB
```

**Cause:** Batch size too large for GPU

**Solution:**
```python
# For testing, use small batch
x = torch.randn(2, 60, 18).cuda()  # batch=2 instead of 64

# For training (Batch 4), use these guidelines:
# RTX 4050 (6GB): batch_size=32 or 64
# RTX 3060 (12GB): batch_size=128
```

---

## 📈 Progress Tracking

### Time Log

| Task | Estimated | Actual | Status |
|------|-----------|--------|--------|
| Setup model directory | 10 min | ___ | ⬜ |
| Test model | 10 min | ___ | ⬜ |
| Run windowing | 45 min | ___ | ⬜ |
| Verify windows | 15 min | ___ | ⬜ |
| Git commit | 10 min | ___ | ⬜ |
| **Total** | **90 min** | ___ | ⬜ |

### Model Testing Log

```
Test 1: Import model
  Status: ___
  Error (if any): ___

Test 2: Forward pass
  Status: ___
  Input shape: ___
  Output shape: ___

Test 3: Loss computation
  Status: ___
  Loss value: ___
```

### Windowing Progress

Track which stocks completed:

```
✅ RELIANCE - 369,017 windows - 280 MB
✅ TCS - 369,009 windows - 275 MB
✅ INFY - 369,012 windows - 278 MB
...
```

---

## 🎯 Ready for Batch 4?

Before proceeding to Batch 4 (Training), ensure:

### Critical
- [x] CARD model tested and working
- [x] All 50 stocks windowed
- [x] Window shapes verified
- [x] No NaN values in windows
- [x] Total ~18.5M windows created

### Recommended
- [x] Model parameters ~2.5M ✅
- [x] Loss function tested
- [x] Window file sizes reasonable (~13.5 GB total)
- [x] Sample windows inspected visually

### Optional
- [ ] Model components split into separate files
- [ ] GPU compatibility tested
- [ ] Window statistics computed
- [ ] Sample predictions generated

---

## 🚀 Next Batch Preview

### Batch 4: Training Pipeline (Final Batch)

**What we'll build:**
1. **DataLoader**
   - Efficient batch loading from .npz files
   - Random sampling across stocks
   - Memory-efficient streaming

2. **Training Loop**
   - Mixed precision training (FP16)
   - Gradient clipping
   - Learning rate scheduling
   - Early stopping

3. **Validation**
   - Direction accuracy (% correct predictions)
   - MAE for returns and volatility
   - Correlation metrics

4. **Checkpointing**
   - Save best model
   - Resume training capability
   - Model versioning

**Files to create:**
- `train/dataset.py` - PyTorch Dataset
- `train/dataloader.py` - Batch loader
- `train/trainer.py` - Training orchestration
- `train/evaluate.py` - Metrics computation
- `train/config.py` - Training configuration
- `train/train.py` - Main training script

**Estimated time:** 4-5 hours
- Code setup: 1 hour
- First training run: 2-3 hours (automated)
- Evaluation: 30 min
- Iteration: 30 min

**Expected results:**
- Direction accuracy: 52-56% (baseline: 50%)
- MAE: 0.003-0.005 (0.3-0.5% per minute)
- Model checkpoint: ~10 MB
- Training time: 2-3 hours on RTX 4050

---

## 📝 Notes & Observations

**Space for your notes:**

**Model testing:**
- Parameters count: ___________
- Forward pass time: ___________
- Memory usage: ___________

**Windowing statistics:**
- Total time: ___________
- Failed stocks: ___________
- Average windows per stock: ___________
- Total data size: ___________

**Data quality observations:**
- Returns range: ___________
- Volatility range: ___________
- Any unexpected patterns: ___________

**Next steps:**
- Ready for Batch 4? ___________
- Any concerns: ___________

---

## 💡 Architecture Deep Dive

### Why 8-minute patches with stride 4?

```
60 minutes input:
[0, 1, 2, 3, 4, 5, 6, 7] → Patch 0 (8 min)
      [4, 5, 6, 7, 8, 9, 10, 11] → Patch 1 (8 min, starts at minute 4)
            [8, 9, 10, 11, 12, 13, 14, 15] → Patch 2 (8 min, starts at minute 8)
            ...
                                    [52, 53, 54, 55, 56, 57, 58, 59] → Patch 12

Total: 13 patches (tokens)
```

**Why this works:**
- 8 minutes captures intraday micro-trends
- Stride 4 = 50% overlap for continuity
- 13 tokens = manageable for attention (not too many)

### Why channel attention before token attention?

1. **Channel Attention** (across 18 features):
   - "Which features matter together?"
   - Example: "High volume + price drop + Nifty drop = strong sell signal"

2. **Token Attention** (across 13 time segments):
   - "Which historical periods matter?"
   - Example: "Last 3 patches (24 min) most important"

**Order matters:** Learn feature interactions FIRST, then temporal patterns.

### Why multi-task loss?

**Returns alone:**
- Model learns: "Price will go up/down"
- No confidence estimate

**Returns + Volatility:**
- Model learns: "Price will go up/down" + "How certain am I?"
- Shared representations improve both
- Volatility helps model learn uncertainty

**Example:**
```
Prediction 1: Return = +0.2%, Volatility = 0.1%
→ Confident upward move

Prediction 2: Return = +0.2%, Volatility = 1.5%
→ Uncertain, avoid trading
```

---

**Batch:** 3 of 4  
**Status:** 🔄 READY TO RUN  
**Updated:** January 31, 2026
