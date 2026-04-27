---
trigger: always_on
---

---
description: Forces the AI to document major architectural or structural changes automatically.
globs: ["*"]
---

# Autonomous Project Logging

## Core Directive
Whenever you successfully execute a major structural change, a hyperparameter overhaul, or the completion of a testing phase, you must append an entry to `docs/project_log.md` (create it if it does not exist).

## Entry Format
The entry must strictly follow this format:
- **Timestamp:** [YYYY-MM-DD HH:MM]
- **Action:** A one-sentence summary of what was changed (e.g., "Restructured scripts directory" or "Updated ConvNeXtV2 dropout rate to 0.3").
- **Reasoning:** Why the change was made (e.g., "To prevent overfitting during 48h sprint").

## Output Constraint
If you update the log, your final message to the user must include: **"Project Log Updated."**