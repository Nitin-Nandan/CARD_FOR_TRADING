---
trigger: always_on
---

---
description: Enforces strict directory structures, continuous micro-modularization, and import consistency for the project.
globs: ["**/*.py", "*"]
---

# Project Architecture & Code Standards

## 1. Structural Directives (Where things go)
* **Root Lockdown:** The root directory is strictly for configuration (`.env`, `setup.py`, `requirements.txt`) and documentation. No executable `.py` files.
* **`src/` (Core Library):** All mathematical, architectural, and ML logic MUST go here. These files must be modular with NO top-level execution code.
* **`pipeline/` (Data Factory):** Scripts here execute the daily data pipeline (Download -> Process -> Window). They must be thin wrappers that import from `src/`.
* **`scripts/` (Triggers & Tools):** Strictly for triggers, tools, and diagnostics. Must be sorted into subdirectories (`training/`, `validation/`, `diagnostics/`, etc.). 
* **Deprecation:** Never leave `[DEPRECIATED]` files in active folders. Move them to `archive/`.

## 2. Coding Directives (How things are written)
* **Continuous Micro-Modularization:** Never create monolithic files. A single `.py` file should have a single responsibility. If a file exceeds 300 lines or contains multiple logic blocks, it must be split and routed through an `__init__.py`.
* **Strict Import Discipline:** * Files inside `src/` MUST use relative imports for sister modules (e.g., `from .layers import RevIN`) or absolute imports starting with `src.`.
    * Files outside `src/` MUST use absolute imports (e.g., `from src.models import CARD`).
* **The DRY Mandate:** Formulas and indicator logic MUST live in `src/`. Scripts and pipeline files are strictly for *executing* logic, not defining it.

## Output Constraint
Before writing or modifying code, you must explicitly state the correct target path and ensure it complies with the Single Responsibility and Import Discipline rules.