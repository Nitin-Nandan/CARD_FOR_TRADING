# Core Project Rules

## 1. PROJECT STRUCTURE — Everything Has a Place

```
project-root/
├── .agents/
│   ├── rules/          ← this file lives here
│   └── skills/         ← agent skill files
├── data/
│   ├── raw/            ← untouched downloads, never modified
│   ├── processed/      ← feature-engineered, split-ready
│   ├── cache/          ← mmap files, intermediate artifacts
│   └── configs/        ← stock universe, date ranges, splits
├── src/
│   ├── data/           ← downloading, parsing, validation
│   ├── features/       ← feature engineering only
│   ├── models/         ← model architecture only (no training logic)
│   ├── training/       ← training loops, schedulers, loss functions
│   ├── evaluation/     ← metrics, backtesting, walk-forward
│   └── utils/          ← shared utilities (logging, config, progress)
├── pipeline/           ← numbered orchestration scripts (00_, 01_, etc.)
├── experiments/        ← one folder per experiment, auto-named
│   └── YYYY-MM-DD_description/
│       ├── config.json
│       ├── metrics.json
│       └── notes.md
├── scripts/            ← one-off utility scripts
├── tests/              ← unit and integration tests
├── docs/               ← project documentation
│   ├── DEVELOPMENT.md
│   ├── EXPERIMENT_REGISTRY.md
│   └── WHAT_WE_KNOW.md
└── archive/            ← old code, never deleted, never imported
```

**Violations of this structure must be flagged before any file is created.**

---

## 2. NO MONOLITHIC CODE

- **One class or one logical function group per file.** If a file exceeds 300 lines, it must be split.
- **No God files.** A file that imports from more than 5 other project files is a warning sign.
- **No logic in `__init__.py`.** These are for imports only.
- **Pipeline scripts (pipeline/XX_name.py) are orchestrators only** — they call functions from `src/`, they contain no business logic themselves.
- **Model architecture files contain zero training logic.** No optimizers, no loss functions, no data loading in `src/models/`.

---

## 3. VISUAL PROGRESS — No Silent Terminals

Every Python script that performs any of the following **must** include a visual progress indicator. Silent terminals are forbidden.

**Mandatory progress for:**
- Any download or API call loop
- Any file read/write loop over multiple files
- Any feature engineering pass over stocks
- Any training epoch
- Any backtesting or evaluation loop

**Standard implementation** — use `tqdm`. Every agent must use this exact pattern:

```python
from tqdm import tqdm

# For stock loops:
for symbol in tqdm(symbols, desc="Processing stocks", unit="stock"):
    ...

# For epoch loops:
for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
    ...

# For file operations:
for file in tqdm(files, desc="Loading data", unit="file"):
    ...
```

**Additionally, every long-running script must print a summary header at start:**
```python
print(f"{'='*60}")
print(f"  Script : {Path(__file__).name}")
print(f"  Task   : [one line description]")
print(f"  Stocks : {len(symbols)}")
print(f"  Start  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
```

**And a completion footer:**
```python
print(f"\n{'='*60}")
print(f"  Done   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Elapsed: {elapsed:.1f}s")
print(f"{'='*60}")
```

---

## 4. CONFIGURATION — No Magic Numbers

- **All hyperparameters, paths, and constants live in config files.** Never hardcoded in scripts.
- Use `src/utils/config.py` with a typed dataclass or Pydantic model.
- Every experiment saves its exact config snapshot to `experiments/YYYY-MM-DD_name/config.json` before running.
- **If a number appears in code that isn't 0 or 1, it belongs in config.**

---

## 5. FINANCIAL ML — Data Integrity Rules

These rules exist because financial ML leakage is silent and lethal.

- **Train/val/test splits are always time-based.** Never random shuffle on time series data.
- **All normalization statistics (mean, std, RevIN params) are computed on training data only**, then applied to val/test. No exceptions.
- **Technical indicators with warm-up periods must have their warm-up rows dropped** before any split boundary.
- **No feature may use future information**, even indirectly (e.g., no volume-weighted average computed over a window that extends past the current bar).
- **Walk-forward validation is the default** for any final evaluation. A single train/test split result is preliminary only.
- Every dataset class must expose a `validate_no_leakage()` method that checks these invariants.

---

## 6. EXPERIMENT TRACKING — Institutional MEMORY

- **Every experiment run must be logged** to `docs/EXPERIMENT_REGISTRY.md` with: date, hypothesis, config hash, key results, conclusion.
- **Negative results are first-class citizens.** A failed experiment with a clear "why it failed" is more valuable than an undocumented success.
- **Never delete experiment folders.** Archive them if needed.
- After any agent completes a task that produces findings, it must update `docs/WHAT_WE_KNOW.md`.

---

## 7. CODE QUALITY — Non-Negotiable

- **All functions have docstrings** with Args and Returns documented.
- **All non-obvious logic has inline comments** explaining *why*, not *what*.
- **Type hints on all function signatures.**
- **No bare `except:` clauses.** Catch specific exceptions.
- **No `print()` for debugging in src/.** Use the project logger (`src/utils/logger.py`).
- `print()` is allowed only in pipeline scripts and only for the header/footer pattern above.

---

## 8. AGENT BEHAVIOUR RULES

- **Agents reason before acting.** No agent modifies or creates a file without first stating its reasoning in a `## Reasoning` block.
- **Agents never execute big scripts.** They write scripts, modify scripts, and give the human clear run instructions. e.g: scripts like downloading, processing and training, etc.
- **Agents flag uncertainty.** If an agent is unsure about a decision, it must say so and present alternatives.
- **Agents update project state after every task.** After completing a task, the agent updates the relevant doc in `docs/` before declaring done.
- **Agents call other agents when out of their domain.** A coding agent that encounters a research question calls the Research Agent rather than guessing.
- **No agent makes assumptions about what the human wants.** If the task is ambiguous, ask one clarifying question before proceeding.

---

## 9. DEPENDENCY MANAGEMENT & CONDA ENVIRONMENT

- All dependencies in `requirements.txt` with pinned versions.
- No new dependency is added without a one-line comment explaining why it's needed.
- `tqdm`, `rich`, `loguru` are standard — always available.
- **Activate the Environment**: Before executing any script or running Python commands, you must activate the conda environment named `card`. For example, use `conda activate card` or prefix commands appropriately (e.g., `conda run -n card <command>`).
- **Handle Missing Dependencies**: If the `card` environment is missing any required dependency, you must **first** add the dependency to the existing `requirements.txt` file in the project. After updating `requirements.txt`, install the dependency **only** into the `card` environment.
- **Strict Isolation**: Nothing should ever be installed into the main system environment. All installations and package modifications must be strictly limited to the `card` conda environment.

---

## 10. AGENT & SKILL MODIFICATION RULES

- Any agent may propose updates to its own `.agents/agents/agent-name.md`
  file or to any skill in `.agents/skills/skill-name/SKILL.md`. It must
  show the exact proposed change and wait for human approval before
  anything is written.

- New agents are created using the agent-development skill.
  New skills are created using the skill-creator skill.
  No agent may create or modify agents/skills by any other method.

- No agent may modify another agent's file directly. It proposes the
  change to the human and the human decides.

- Self-improvement proposals happen at the END of a task only.
  Never mid-task.

- Human approval is required before any agent or skill file is
  written, updated, or created. No exceptions.

