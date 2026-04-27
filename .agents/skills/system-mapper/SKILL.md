---
name: system-mapper
description: "MANDATORY: Maintain the living architectural blueprint (docs/SYSTEM_MAP.md) whenever a Python file is created, deleted, moved, or when its inputs/outputs fundamentally change. Use this skill to keep the system map up-to-date as a single source of truth for the data flow and file responsibilities across src/, pipeline/, and scripts/."
risk: low
source: custom
date_added: "2026-03-23"
---

# System Mapper Skill

> **Core Directive:** This skill maintains a single living document: `docs/SYSTEM_MAP.md`. It must be activated whenever a Python file is **created**, **deleted**, **moved**, or when its **inputs/outputs are fundamentally changed**.

## When to Trigger

- A new `.py` file is added to `src/`, `pipeline/`, or `scripts/`.
- An existing `.py` file is deleted or moved between directories.
- A file's core responsibility changes (e.g., it now consumes a new data source or produces a new artifact).
- After a major structural refactor or reorganization of the project.

## What the Map Contains

The `docs/SYSTEM_MAP.md` file uses **Markdown tables only** — no raw code. It tracks metadata for every active Python file in the project's three core zones:

1. **`src/` (Core Library)** — The engine. Pure logic, no execution.
2. **`pipeline/` (Data Factory)** — Sequential steps that transform raw data into training-ready windows.
3. **`scripts/` (Triggers & Tools)** — Entrypoints, diagnostics, and utilities.

## Table Schema

Each table row must contain exactly these 5 columns:

| Column | Description |
|---|---|
| **Filepath** | Relative path from project root (e.g., `src/data/builder.py`) |
| **Core Purpose** | A brief, one-line description of the file's responsibility |
| **Inputs** | What data, tensors, files, or configs it consumes |
| **Outputs** | What data, tensors, files, or artifacts it produces |
| **Dependencies** | Which other project files import or rely on this file |

## Rules

1. **Never duplicate code.** The map tracks *metadata*, not implementation.
2. **One row per file.** If a file has multiple responsibilities, summarize them.
3. **Keep descriptions terse.** Use shorthand like `(N, 81, 60) tensor` or `.parquet files`.
4. **Organize by zone.** Use separate tables for `src/`, `pipeline/`, and `scripts/`.
5. **Include `__init__.py` only if it re-exports symbols.**

## Procedure

1. **Read** the current `docs/SYSTEM_MAP.md`.
2. **Identify** which files were affected by the change.
3. **Update** the relevant rows (add, remove, or modify).
4. **Verify** that the Dependencies column is consistent (if file A imports file B, both rows should reflect that relationship).
5. **Save** and confirm the update.

## Example Row

| Filepath | Core Purpose | Inputs | Outputs | Dependencies |
|---|---|---|---|---|
| `src/data/builder.py` | Creates sliding windows from processed stock data | `.parquet` files from `data/processed/` | `_windows.npz` files in `data/windows/` | `pipeline/03_windowing.py` |
