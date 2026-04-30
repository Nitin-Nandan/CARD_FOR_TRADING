# Conda Environment Usage Rule

Whenever you need to execute any script or run Python commands, you MUST follow these strict guidelines:

1. **Activate the Environment**: Before executing any script, you must activate the conda environment named `card`. For example, use `conda activate card` or prefix your commands appropriately (e.g., `conda run -n card <command>`).
2. **Handle Missing Dependencies**:
   - If the `card` environment is missing any required dependency, you must **first** add the dependency to the existing `requirements.txt` file in the project.
   - After updating `requirements.txt`, install the dependency **only** into the `card` environment.
3. **Strict Isolation**: Nothing should ever be installed into the main system environment. All installations and package modifications must be strictly limited to the `card` conda environment.
