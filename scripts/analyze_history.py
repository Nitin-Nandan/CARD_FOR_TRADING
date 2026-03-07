import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR  = PROJECT_ROOT / "logs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def summarize(json_path):
    with open(json_path) as f:
        d = json.load(f)
        
    out_path = REPORTS_DIR / "analysis_output.txt"
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write("==================================================\n")
        out.write(f"ANALYSIS OF {json_path}\n")
        
        tl = d['train_loss']
        vl = d['val_loss']
        epochs = len(tl)
        out.write(f"Epochs trained: {epochs}\n")
        
        out.write("\n[LOSS (Normalized MAE)]\n")
        bvl = min(vl)
        bvl_e = vl.index(bvl) + 1
        out.write(f"Val Loss  : start={vl[0]:.5f} -> end={vl[-1]:.5f} | BEST: {bvl:.5f} at E{bvl_e}\n")
        btl = min(tl)
        btl_e = tl.index(btl) + 1
        out.write(f"Train Loss: start={tl[0]:.5f} -> end={tl[-1]:.5f} | BEST: {btl:.5f} at E{btl_e}\n")
        
        out.write("\n[METRICS]\n")
        for m in ['mae', 'direction_accuracy', 'correlation']:
            tv = [e[m] for e in d['train_metrics']]
            vv = [e[m] for e in d['val_metrics']]
            if 'accuracy' in m or 'correlation' in m:
                out.write(f"Val {m:18s}: start={vv[0]:.2f} -> best={max(vv):.2f} (E{vv.index(max(vv))+1})\n")
                out.write(f"Trn {m:18s}: start={tv[0]:.2f} -> best={max(tv):.2f} (E{tv.index(max(tv))+1})\n")
            else:
                out.write(f"Val {m:18s}: start={vv[0]:.2f} -> best={min(vv):.2f} (E{vv.index(min(vv))+1})\n")
                out.write(f"Trn {m:18s}: start={tv[0]:.2f} -> best={min(tv):.2f} (E{tv.index(min(tv))+1})\n")

if __name__ == '__main__':
    summarize(sys.argv[1])
