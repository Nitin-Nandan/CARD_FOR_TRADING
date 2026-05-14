---
name: project-tidy
description: Audits and tidies the NSE CARD project file structure. Triggers when the human says "tidy the project", "clean up structure", "reorganise files", "audit the layout", "files are in the wrong place", or after any large coding session. Reads the full project tree, classifies every file against the canonical structure and hygiene rules, writes a TIDY PLAN with exact proposed moves/deletions, waits for human approval, then executes approved changes one at a time. Proactively identifies and proposes deletion of __pycache__, redundant .gitkeep, and obsolete archive contents. Use this skill proactively after any session that creates multiple new files.
---

# Project Tidy Skill

This skill is used to audit and clean up the structure of the NSE stock prediction project (CARD ICLR'24) without breaking anything.

## When to Use This Skill
- Human says "tidy the project", "clean up the structure", "reorganise the files", "the project structure is a mess", "sort out the folders"
- Human asks "does this project follow the rules?", "is the structure correct?", "audit the project layout"
- Human says "I've added some new files, can you put them in the right place"
- After any large coding session where multiple files may have been created in the wrong locations
- When onboarding a new session and the human wants to re-establish structure

## Canonical Project Structure
This is the target structure from the modular rules (specifically `structure.md` and `hygiene.md`). You must measure every file against this:
```text
project-root/
├── .agents/
│   ├── rules/          ← rules files only
│   └── skills/         ← skill folders only
├── data/
│   ├── raw/            ← untouched downloads, never modified
│   ├── processed/      ← feature-engineered, split-ready data
│   ├── cache/          ← mmap files, intermediate artifacts
│   └── configs/        ← stock universe, date ranges, splits
├── src/
│   ├── data/           ← downloading, parsing, validation only
│   ├── features/       ← feature engineering only
│   ├── models/         ← model architecture only (ZERO training logic)
│   ├── training/       ← training loops, schedulers, loss functions
│   ├── evaluation/     ← metrics, backtesting, walk-forward
│   └── utils/          ← config, logger, progress utilities
├── pipeline/           ← numbered orchestration scripts only (00_, 01_, etc.)
├── experiments/        ← one folder per experiment, auto-named YYYY-MM-DD_desc/
│   └── YYYY-MM-DD_name/
│       ├── config.json
│       ├── metrics.json
│       └── notes.md
├── scripts/            ← one-off utility and helper scripts
├── tests/              ← unit and integration tests only
├── docs/               ← project documentation
│   ├── DEVELOPMENT.md
│   ├── EXPERIMENT_REGISTRY.md
│   └── WHAT_WE_KNOW.md
└── archive/            ← old code only, never imported by anything active
```

## Rules You MUST Follow During Audit

### NEVER Touch These (PROTECTED)
- Any file inside `data/raw/` — raw data is sacred, do not move or rename
- Any file inside `data/cache/` — mmap and cache files have hardcoded paths in code, moving them breaks things silently
- Any `.env` file anywhere
- Any `*.pth`, `*.pt`, `*.ckpt`, `*.pkl` files — trained model weights
- Any `*.parquet`, `*.csv`, `*.feather` files inside `data/` — processed datasets
- The `.git/` directory obviously
- Any file the human has explicitly told you to leave alone in this session

### ALWAYS Flag These as HYGIENE VIOLATIONS (SAFE to propose deletion)
- `__pycache__/` folders or `*.pyc` files — forbidden in the workspace.
- `.gitkeep` files in directories that already contain other files — redundant and messy.
- Obsolete or empty files/directories in `archive/` — keep only relevant legacy reference.

### ALWAYS Flag These as VIOLATIONS
Do NOT auto-fix these; require a human decision:
- A Python file in `src/models/` that contains a training loop, optimizer, or loss function — this is a boundary violation (model files must be pure architecture)
- A Python file in `pipeline/` that contains business logic beyond function calls — should only orchestrate `src/`
- Any Python file longer than 300 lines — flag it, suggest split point, do not split automatically
- A file named `utils.py` or `helpers.py` or `misc.py` at the project root — these are anti-patterns
- Any import of `archive/` from active code — dead code being used is dangerous
- Duplicate filenames in different locations that appear to serve the same purpose — flag for human to decide which is canonical
- Any hardcoded path string (not from config) in a Python file in `src/` — flag the file and line number

### SAFE to Move Automatically (Proposed Moves)
You may propose to move these (after plan approval):
- Python files sitting at project root that clearly belong in `src/` subpackages
- Experiment folders not inside `experiments/` (e.g., a folder named `run_2024_03/` at root)
- `*.log` files at project root → move to `logs/` (create if needed)
- Test files (`test_*.py`) sitting in `src/` → move to `tests/`
- A script named `train_*.py` sitting at root → propose moving to `pipeline/` or `scripts/`
- Documentation files (`.md`) at root that are not `README.md` → propose moving to `docs/`

### STRICT PROHIBITIONS
- **NEVER rename files — only move them.** Renaming breaks imports. Only move files to correct locations. If a file needs renaming, flag it to the human with a suggestion but do not execute the rename.
- **NEVER execute any Python script or shell command** (other than moving files).
- **NEVER delete any core project file.** Only hygiene violations (pycache, redundant gitkeep, obsolete archive) can be proposed for deletion.
- **NEVER touch raw data or model weights.**
- **NEVER move anything without showing the plan and getting approval first.**
- **NEVER update import paths automatically** — only list them for the human.
- **NEVER make assumptions about what a file does** without reading at least its first 20 lines and its filename.

---

## Step-by-Step Workflow

### Step 1: Announce and scan
1. Say: "Starting project structure audit. Reading current file tree..."
2. Read the full directory tree from project root. Skip `data/raw/`, `data/cache/`, `.git/`, `__pycache__/`, `node_modules/` in the display (they're too noisy) but note how many files are in each.

### Step 2: Classify every file
For each file found, silently classify it as one of:
- **CORRECT** — already in the right place
- **MISPLACED** — should be somewhere else (safe to propose move)
- **HYGIENE** — forbidden/redundant file (safe to propose deletion)
- **PROTECTED** — in the never-touch list (skip entirely)

### Step 3: Write the TIDY PLAN
Present a structured plan to the human **BEFORE** doing anything. Format exactly like this:

```markdown
## TIDY PLAN
Audit complete. Here is what I found and propose.

### Summary
- Files scanned: N
- Correct: N
- Proposed moves: N  
- Violations to review: N
- Unknown (need your input): N

---

### Proposed Moves (safe to execute after your approval)

1. MOVE `root/train_card.py` → `pipeline/train_daily.py`
   Reason: Training orchestration script sitting at root. No logic change, import paths unaffected as it's a pipeline script.

2. DELETE `src/data/__pycache__`
   Reason: Forbidden temporary directory.

3. DELETE `logs/.gitkeep`
   Reason: Directory is no longer empty; .gitkeep is redundant.

4. DELETE `archive/intraday_v1/empty_subdir`
   Reason: Empty and obsolete archive content.

[... and so on for each proposed move ...]

---

### Violations (require your decision — I will NOT touch these automatically)

1. VIOLATION: `src/models/card_model.py` contains an optimizer instantiation on line 187
   Rule broken: Model files must be pure architecture, no training logic
   Suggestion: Extract the optimizer to `src/training/trainer.py`
   Your options: (a) I can help you refactor this, (b) ignore this violation, (c) flag for later

2. VIOLATION: `src/features/engine.py` is 412 lines (limit: 300)
   Suggestion: Natural split point at line 203 where volatility features begin — could become `src/features/volatility.py`
   Your options: (a) I can plan the split for you, (b) ignore for now

[... and so on ...]

---

### Unknown — I need your input

1. UNKNOWN: `misc_experiments/` folder at root — 3 Python files inside
   What is this? Options: (a) move to `archive/`, (b) move to `experiments/`, (c) leave it

---

Do you approve the Proposed Moves? Say "proceed" to execute them, or tell me which ones to skip.
Violations and Unknowns will NOT be touched until you give specific instructions for each.
```

### Step 4: Wait for human response
Do not proceed until the human explicitly approves. If they say "proceed", execute all proposed moves. If they say "skip 2 and 4, proceed with rest", honour that exactly.

### Step 5: Execute approved moves one at a time
For each approved move, say exactly what you are doing before doing it:
`Moving src/data/test_dataset.py → tests/test_dataset.py ... done.`
After each move, verify the file exists at the new location before announcing done.
If any move fails for any reason — stop immediately and report the error. Do not continue with remaining moves until the human acknowledges.

### Step 6: Check for broken imports (important)
After all moves are complete, scan every Python file that imports from a moved file and check if the import path needs updating. Do not update imports automatically — list them for the human:

```markdown
## Import paths to update after moves

The following files import from locations that have changed:
1. `src/training/trainer.py` line 3: `from src.data.test_dataset import ...`
   → Update to: `from tests.test_dataset import ...`
   (Note: test imports in src/ are unusual — verify this is intentional)
```

### Step 7: Write TIDY REPORT
Save to `docs/TIDY_REPORT_YYYY-MM-DD.md` (replace YYYY-MM-DD with today's date):

```markdown
# Project Tidy Report
**Date**: YYYY-MM-DD
**Files scanned**: N
**Moves executed**: N
**Violations flagged**: N (N resolved, N deferred)

## Moves Executed
| From | To | Status |
|------|-----|--------|

## Violations Flagged
| File | Rule | Status |
|------|------|--------|

## Import Paths Updated
| File | Old import | New import |
|------|-----------|-----------|

## Deferred Items
[Anything the human said to handle later]
```

Then update `docs/WHAT_WE_KNOW.md` with a one-liner: "Project structure audited on YYYY-MM-DD. N files moved. N violations outstanding."
