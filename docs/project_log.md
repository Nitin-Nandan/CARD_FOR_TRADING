# Project Log

- **Timestamp:** 2026-03-20 04:26
- **Action:** Restructured scripts/ and data/ directories and archived deprecated files.
- **Reasoning:** To resolve file/directory name conflicts and enforce strict subdomain organization for better project scalability.

- **Timestamp:** 2026-03-20 05:40
- **Action:** Restored and modularized the data pipeline into `src/` and created thin `pipeline/` entrypoints.
- **Reasoning:** To complete the decoupling of the legacy scripts and formalize the data processing library.

- **Timestamp:** 2026-03-20 05:55
- **Action:** Executed The Final Purge. Deleted legacy `models/`, `train/`, `utils/`, and `_OLD_APPROACH_BACKUP/`.
- **Reasoning:** To reach the final clean state and ensure `src/` is the single source of truth, following strict preservation of data and weights.

- **Timestamp:** 2026-03-20 06:25
- **Action:** Consolidated Phase 0-4 logs into `docs/DEVELOPMENT.md` and purged legacy docs.
- **Reasoning:** To streamline project documentation and move historical logs into a single developmental source of truth.

- **Timestamp:** 2026-03-20 07:25
- **Action:** Refactored `src/models/card.py` into modular components (`layers.py`, `attention.py`, `loss.py`) without logic mutation.
- **Reasoning:** To enforce strict Single Responsibility and improve maintainability.

- **Timestamp:** 2026-03-20 08:00
- **Action:** Relocated custom `pandas_ta` fork to `vendor/` directory for architectural compliance.
- **Reasoning:** To segregate external dependencies from core source code.

- **Timestamp:** 2026-03-20 08:10
- **Action:** Finalized Phase 5 Production README and locked architectural standards.
- **Reasoning:** To formalize the Project CARD documentation and provide a clear system architecture roadmap for future development.

- **Timestamp:** 2026-03-23 15:00
- **Action:** Ran global lint-and-validate suite. Fixed all syntax and formatting violations.
- **Reasoning:** To enforce code quality standards across the entire codebase. Fixed 22 ruff violations including 4 critical undefined name errors in `src/data/dataset.py`, 2 bare excepts, 1 unused variable, 1 ambiguous variable name, 1 late import, and 12 E402 annotations. All 50 Python files now pass ruff check and compileall with zero errors.

- **Timestamp:** 2026-03-23 15:19
- **Action:** Created `system-mapper` skill and bootstrapped `docs/SYSTEM_MAP.md` with metadata for all 31 active Python files.
- **Reasoning:** To maintain a living architectural blueprint as a single source of truth for file responsibilities, data flow, and inter-file dependencies across `src/`, `pipeline/`, and `scripts/`.
