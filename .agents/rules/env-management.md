---
trigger: always_on
---

---
description: Ensures all execution happens within the 'card' conda environment on a Windows/PowerShell host.
globs: ["**/*.py", "requirements.txt"]
---

# Environment & Windows Execution Standards

## Core Directives
1. **Windows Native Execution:** This project is hosted on **Windows**. You MUST use **PowerShell 7+** syntax for all terminal operations. 
   - Never use `ls`, `rm`, `cp`, or `export`. 
   - Use `Get-ChildItem`, `Remove-Item -Recurse -Force`, `Copy-Item`, and `$env:VAR = "val"`.
2. **The 'Card' Mandate:** Every Python execution MUST be preceded by the Conda activation command for Windows.
   - **Correct Syntax:** `conda activate card; python path\to\script.py`
3. **Path Handling:** Always use backslashes `\` for shell commands, but ensure Python scripts handle paths using `pathlib` to maintain cross-compatibility.
4. **Dependency Failure Recovery:** - **Step A:** If a script fails with `ModuleNotFoundError`, verify the `card` env is active.
   - **Step B:** If missing, append the package to `requirements.txt` and run `pip install` or `conda install` within the active `card` session.

## Output Constraint
If a command is corrected from Linux to PowerShell, the agent must include: **"Windows/PowerShell syntax enforced."**