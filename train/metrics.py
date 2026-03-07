"""
Evaluation Metrics for CARD Close-Price Prediction

Metrics:
1. Direction Accuracy  — % correct up/down between consecutive predicted steps
   (np.sign(price) is always +1 for INR, so we use step-over-step delta instead)
2. MAE  — Mean Absolute Error in ₹
3. RMSE — Root Mean Squared Error in ₹
4. Correlation — Pearson between pred and target trajectories
"""

import torch
import numpy as np
from typing import Dict, Tuple


class MetricsCalculator:
    """
    Accumulates batches of (pred_close, y_close) tensors, then computes:
      - direction_accuracy : step-over-step change direction match (%)
      - mae                : Mean Abs Error in ₹
      - rmse               : Root Mean Squared Error in ₹
      - correlation        : Pearson r between flattened pred & target
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.all_preds   = []
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            predictions: (B, pred_len) — predicted close prices
            targets:     (B, pred_len) — actual close prices
        """
        preds = predictions.detach().cpu().numpy()   # (B, T)
        targs = targets.detach().cpu().numpy()       # (B, T)
        self.all_preds.append(preds)
        self.all_targets.append(targs)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics over accumulated batches."""
        if not self.all_preds:
            return {}

        # Stack: shape (N_total_batches, B, T)
        preds   = np.concatenate(self.all_preds,   axis=0)   # (N, T)
        targets = np.concatenate(self.all_targets, axis=0)   # (N, T)
        N, T    = preds.shape

        metrics = {}

        # ====================================================================
        # DIRECTION ACCURACY
        # For close prices, sign(price) is always +1.
        # Instead: compare step-over-step changes within the 15-step window.
        # T-1 deltas per sample.  If T==1 this is undefined (skip).
        # ====================================================================
        if T > 1:
            pred_delta   = np.diff(preds,   axis=1)  # (N, T-1)
            target_delta = np.diff(targets, axis=1)  # (N, T-1)
            dir_correct  = (np.sign(pred_delta) == np.sign(target_delta)).astype(float)
            metrics['direction_accuracy'] = dir_correct.mean() * 100

            up_mask   = target_delta > 0
            down_mask = target_delta < 0
            if up_mask.sum() > 0:
                metrics['direction_accuracy_up']   = dir_correct[up_mask].mean()   * 100
            if down_mask.sum() > 0:
                metrics['direction_accuracy_down'] = dir_correct[down_mask].mean() * 100

        # ====================================================================
        # REGRESSION METRICS  (flat over all N*T values)
        # ====================================================================
        p_flat = preds.reshape(-1)
        t_flat = targets.reshape(-1)

        mae  = np.abs(p_flat - t_flat).mean()
        mse  = ((p_flat - t_flat) ** 2).mean()
        rmse = np.sqrt(mse)

        metrics['mae']  = mae
        metrics['mse']  = mse
        metrics['rmse'] = rmse

        # ====================================================================
        # CORRELATION  (on flattened arrays)
        # ====================================================================
        if p_flat.std() > 0 and t_flat.std() > 0:
            metrics['correlation'] = float(np.corrcoef(p_flat, t_flat)[0, 1])
        else:
            metrics['correlation'] = 0.0

        return metrics
    
    def compute_and_reset(self) -> Dict[str, float]:
        """Compute metrics and reset"""
        metrics = self.compute()
        self.reset()
        return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    if not metrics:
        return f"{prefix}No metrics"
    lines = []
    if 'direction_accuracy' in metrics:
        lines.append(f"{prefix}Dir Acc: {metrics['direction_accuracy']:.1f}%")
        if 'direction_accuracy_up' in metrics:
            lines.append(
                f"  ↑ Up: {metrics['direction_accuracy_up']:.1f}%  "
                f"↓ Down: {metrics.get('direction_accuracy_down', 0):.1f}%"
            )
    if 'mae' in metrics:
        lines.append(
            f"{prefix}MAE: ₹{metrics['mae']:.2f}  RMSE: ₹{metrics.get('rmse', 0):.2f}"
        )
    if 'correlation' in metrics:
        lines.append(f"{prefix}Correlation: {metrics['correlation']:.4f}")
    return "\n".join(lines)


def print_metrics_table(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    epoch: int = None
):
    """
    Print metrics in a nice table format
    
    Args:
        train_metrics: Training metrics
        val_metrics: Validation metrics
        epoch: Current epoch number
    """
    header = "="*70
    print(header)
    if epoch is not None:
        print(f"EPOCH {epoch} RESULTS")
        print(header)
    
    # Get all metric names
    all_keys = set(train_metrics.keys()) | set(val_metrics.keys())
    
    # Sort: direction accuracy first, then alphabetically
    priority_keys = ['direction_accuracy', 'mae', 'mse', 'correlation']
    other_keys = sorted([k for k in all_keys if k not in priority_keys])
    ordered_keys = [k for k in priority_keys if k in all_keys] + other_keys
    
    # Print header
    print(f"{'Metric':<30} {'Train':>15} {'Val':>15}")
    print("-"*70)
    
    # Print metrics
    for key in ordered_keys:
        train_val = train_metrics.get(key, 0)
        val_val = val_metrics.get(key, 0)
        
        # Format based on metric type
        if 'accuracy' in key or 'correlation' in key:
            print(f"{key:<30} {train_val:>15.2f} {val_val:>15.2f}")
        else:
            print(f"{key:<30} {train_val:>15.6f} {val_val:>15.6f}")
    
    print(header)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Metrics Calculator...")
    print("="*60)
    
    # Create fake predictions and targets
    batch_size = 32
    pred_len = 15
    
    # Simulate predictions with 55% direction accuracy
    np.random.seed(42)
    targets = np.random.randn(batch_size, pred_len) * 0.01
    
    # Predictions: 55% correct direction, some noise
    predictions = np.zeros_like(targets)
    for i in range(batch_size):
        for j in range(pred_len):
            if np.random.rand() < 0.55:
                # Correct direction
                predictions[i, j] = targets[i, j] + np.random.randn() * 0.003
            else:
                # Wrong direction
                predictions[i, j] = -targets[i, j] + np.random.randn() * 0.003
    
    # Convert to tensors
    pred_tensor = torch.from_numpy(predictions).float()
    target_tensor = torch.from_numpy(targets).float()
    
    # Calculate metrics
    calculator = MetricsCalculator()
    calculator.update(pred_tensor, target_tensor)
    metrics = calculator.compute()
    
    # Print
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*60)
    print("Formatted output:")
    print("="*60)
    print(format_metrics(metrics, prefix="Test: "))
    
    print("\n" + "="*60)
    print("✅ Metrics calculator test complete!")
    print("="*60)
