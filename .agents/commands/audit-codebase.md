---
description: Performs a comprehensive code review, removes legacy comments, and verifies logic integrity for a specified target.
globs: ["**/*.py"]
---

# Codebase Audit & Cleanup Protocol

## Core Directives
1. **Legacy Scrubbing:** Remove all outdated multiline comments, "Phase 0/1/2/3" notes, and commented-out legacy code (e.g., old scale_factor hacks). Keep only clean, professional docstrings that explain current functionality.
2. **Bug & Syntax Verification:** Scan the code for undefined variables, shape mismatch risks in tensors, and missing imports.
3. **Import Alignment:** Ensure all imports strictly follow the `architecture-and-standards.md` rule (absolute imports from `src.` for external scripts, relative imports within `src/`).
4. **Execution Integrity:** Do not alter the core mathematical logic or tensor operations during the cleanup. Your goal is formatting and safety, not rewriting the math.

## Output Constraint
You must provide a brief summary of the bugs found and the legacy comments removed before writing the cleaned code.