# Project Hygiene

## 1. Forbidden Files
- **`__pycache__`**: `__pycache__` folders and `.pyc` files are strictly forbidden. Agents MUST delete them immediately if found.
- **`.gitkeep`**:
  - PERMITTED: In mandatory empty directories (e.g., `logs/`, `checkpoints/`) during fresh setup.
  - FORBIDDEN: In any directory that contains at least one other file.

## 2. Archive Management
- The `archive/` folder is for legacy code reference only.
- If a folder or file in `archive/` is empty or its purpose is no longer valid, it MUST be deleted.

## 3. Tool Utilization
- Use the `project-tidy` skill for structural cleanup instead of writing manual scripts.
- If `project-tidy` lacks a feature, update the skill via `skill-creator`.
