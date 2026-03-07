"""Quick pre-training data audit: checks X is raw (not normalized) and has no NaN/Inf."""
import sys
from pathlib import Path
import numpy as np

sys.stdin = open('nul')   # avoid interactive prompts

WINDOWS_DIR = Path(__file__).parent.parent / "data" / "windows"

def check_stock(name):
    f = WINDOWS_DIR / f"{name}_windows.npz"
    if not f.exists():
        print(f"  MISSING: {f}")
        return False
    d  = np.load(f, allow_pickle=True)
    X  = d['X'];  yc = d['y_close'];  ci = int(d['close_idx'])
    cv = X[:, :, ci]
    nan_X = np.isnan(X).sum();   inf_X = np.isinf(X).sum()
    nan_y = np.isnan(yc).sum();  inf_y = np.isinf(yc).sum()
    raw   = cv.max() > 50        # raw ₹ prices always >> 50

    status = "✅" if raw and nan_X == 0 and inf_X == 0 and nan_y == 0 and inf_y == 0 else "❌"
    print(f"  {status} {name:12s}  close=[{cv.min():.1f},{cv.max():.1f}]  "
          f"y=[{yc.min():.1f},{yc.max():.1f}]  "
          f"NaN_X={nan_X}  NaN_y={nan_y}  Inf_X={inf_X}  raw={'YES' if raw else 'NO (NORMALIZED!)'}")
    return raw and nan_X == 0

CHECK = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "ITC",
         "SBIN", "WIPRO", "INFY", "AXISBANK", "BAJFINANCE"]

print("="*70)
print("DATA AUDIT — checking X is raw + no NaN/Inf")
print("="*70)
ok = sum(check_stock(s) for s in CHECK)
print(f"\n{ok}/{len(CHECK)} stocks passed ✅")
if ok < len(CHECK):
    print("⚠️  Some stocks FAILED — re-run pipeline/03_create_windows_60min.py")
else:
    print("All good — safe to start training.")
