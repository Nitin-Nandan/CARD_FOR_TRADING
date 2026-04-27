"""
Weights & Biases Logger Custom Wrapper
"""

import os

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger:
    def __init__(
        self,
        project,
        name,
        config=None,
        entity=None,
        resume=False,
        tags=None,
        notes=None,
        offline=False,
    ):
        """
        Initialize WandB logger

        Args:
            project: WandB project name (e.g., 'card-stock-prediction')
            name: Run name (e.g., 'phase4-final-training')
            config: Dict of hyperparameters to log
            entity: WandB username/team (optional, uses default)
            resume: Resume run if it exists (default False)
            tags: List of tags for organization (e.g., ['phase4', 'final'])
            notes: Text description of the run
            offline: If True, saves logs locally and syncs later (default False)
                     USE THIS IF INTERNET IS UNRELIABLE
        """

        if not WANDB_AVAILABLE:
            self.enabled = False
            print("⚠️  WandB logging disabled (wandb package not installed)")
            return

        self.enabled = True

        # Set offline mode if internet is unreliable
        if offline:
            os.environ["WANDB_MODE"] = "offline"
            print("⚠️  WandB running in OFFLINE mode")
            print("   Logs saved locally, will sync when online")

        # Initialize run
        try:
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                entity=entity,
                resume="allow" if resume else False,
                tags=tags or [],
                notes=notes or "",
            )

            if offline:
                print("✓ WandB initialized (OFFLINE)")
                print("  Logs: ~/.wandb/ (sync with 'wandb sync')")
            else:
                print("✓ WandB initialized (ONLINE)")
                print(f"  Dashboard: {self.run.url}")

        except Exception as e:
            print(f"⚠️  WandB initialization failed: {e}")
            print("   Falling back to offline mode...")
            os.environ["WANDB_MODE"] = "offline"

            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                resume=False,
                tags=tags or [],
                notes=notes or "",
            )

            print("✓ WandB initialized (OFFLINE - fallback)")

    def log_metrics(self, metrics, step=None):
        """Log a dictionary of metrics"""
        if not self.enabled:
            return

        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def log_model(self, model_path, name="model"):
        """Save and log a model checkpoint"""
        if not self.enabled:
            return

        # Create an artifact
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(model_path)

        # Log it
        self.run.log_artifact(artifact)
        print(f"✓ Logged model artifact: {name}")

    def close(self):
        """Finish the run"""
        if self.enabled:
            wandb.finish()
