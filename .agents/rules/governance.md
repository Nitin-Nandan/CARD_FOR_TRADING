# Agent Governance

## 1. Skill Evolution (Rule 10)
- If an agent performs a manual task that should be automated or repeated, it MUST propose a new skill or update an existing one.
- Never write "temporary scripts" for tasks that fall under project infrastructure (like cleanup).

## 2. Decision Logic
- **Scientific Rationale**: Every implementation change must be preceded by a "reasoning" block or a research subagent report.
- **Human Approval**: Mandatory for any skill update, project rule change, or structural move.

## 3. Code Quality
- Files MUST NOT exceed 300 lines.
- Modularity is strictly enforced. Boundary violations (e.g., training logic in architecture files) are forbidden.
