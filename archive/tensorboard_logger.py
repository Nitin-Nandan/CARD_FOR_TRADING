"""
TensorBoard Logger Custom Wrapper
"""

import os
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TensorBoardLogger:
    def __init__(self, log_dir, name):
        """
        Initialize TensorBoard logger

        Args:
            log_dir: Base directory for logs
            name: Run name
        """
        if not TENSORBOARD_AVAILABLE:
            self.enabled = False
            print("⚠️  TensorBoard logging disabled (tensorboard package not installed)")
            return

        self.enabled = True

        # Create run-specific directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(log_dir, f"{name}_{timestamp}")

        os.makedirs(run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=run_dir)
        print(f"✓ TensorBoard initialized: {run_dir}")
        print(f"  To view: tensorboard --logdir={log_dir}")

    def log_metrics(self, metrics, step):
        """Log a dictionary of metrics at a specific step"""
        if not self.enabled:
            return

        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        """Close the writer"""
        if self.enabled:
            self.writer.close()
