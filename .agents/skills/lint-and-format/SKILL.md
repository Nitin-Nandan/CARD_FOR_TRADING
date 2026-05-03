---
name: lint-and-format
description: >
  Code formatting and linting skill. Uses Ruff and Black to enforce
  PEP-8 compliance and codebase cleanliness. Triggers when human asks to
  "format the code", "run the linter", "fix formatting", or "clean up code".
  Always ensures files remain under 300 lines by warning if they exceed it.
---

# Lint and Format Skill

## Objective
Automatically format Python code using Black and lint using Ruff. Warn the human
if any single file exceeds 300 lines, enforcing the non-negotiable rule.

## Execution Steps

1. Run Black for formatting:
   `black src/ pipeline/ tests/`
2. Run Ruff for linting and automatic fixes:
   `ruff check --fix src/ pipeline/ tests/`
3. Scan for monolithic files:
   `find src/ pipeline/ -name "*.py" -exec wc -l {} + | awk '$1 > 300 {print}'`
4. If any file exceeds 300 lines, output a prominent WARNING to the human and list the files.
