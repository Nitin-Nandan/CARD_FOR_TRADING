# Graphify Exclusions Rule

When using the `/graphify` skill or running the graphify pipeline:
- **NEVER** include or process the `vendor` directory.
- The `vendor` directory contains third-party dependencies and should always be excluded to save tokens and prevent unnecessary noise in the knowledge graph.
- This is enforced by the `.graphifyignore` file, which must contain `vendor/`. Do not remove it.
