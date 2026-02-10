"""
Evaluation Metrics for Stock Prediction

Comprehensive metrics to evaluate model performance:
- Direction accuracy (most important for trading)
- MAE / MSE (standard regression metrics)
- Correlation (prediction quality)
- Sharpe-like ratio (risk-adjusted returns)
"""

import torch
import numpy as np
from typing import Dict, Tuple


class MetricsCalculator:
    """
    Calculate all evaluation metrics
    
    Metrics:
    1. Direction Accuracy: % of correct up/down predictions
    2. MAE: Mean Absolute Error
    3. MSE: Mean Squared Error  
    4. RMSE: Root Mean Squared Error
    5. Correlation: Pearson correlation
    6. Sharpe: Risk-adjusted returns metric
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.all_preds = []
        self.all_targets = []
        self.all_volatility_preds = []
        self.all_volatility_targets = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        volatility_preds: torch.Tensor = None,
        volatility_targets: torch.Tensor = None
    ):
        """
        Accumulate predictions and targets
        
        Args:
            predictions: (batch, pred_len) or (batch, channels, pred_len)
            targets: Same shape as predictions
            volatility_preds: Optional volatility predictions
            volatility_targets: Optional volatility targets
        """
        # Move to CPU and convert to numpy
        preds = predictions.detach().cpu().numpy()
        targs = targets.detach().cpu().numpy()
        
        # Flatten if multi-channel
        if preds.ndim == 3:
            # (batch, channels, pred_len) -> (batch * channels * pred_len,)
            preds = preds.reshape(-1)
            targs = targs.reshape(-1)
        elif preds.ndim == 2:
            # (batch, pred_len) -> (batch * pred_len,)
            preds = preds.reshape(-1)
            targs = targs.reshape(-1)
        
        self.all_preds.append(preds)
        self.all_targets.append(targs)
        
        # Volatility
        if volatility_preds is not None and volatility_targets is not None:
            vol_preds = volatility_preds.detach().cpu().numpy()
            vol_targs = volatility_targets.detach().cpu().numpy()
            
            if vol_preds.ndim == 3:
                vol_preds = vol_preds.reshape(-1)
                vol_targs = vol_targs.reshape(-1)
            elif vol_preds.ndim == 2:
                vol_preds = vol_preds.reshape(-1)
                vol_targs = vol_targs.reshape(-1)
            
            self.all_volatility_preds.append(vol_preds)
            self.all_volatility_targets.append(vol_targs)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary of metric name -> value
        """
        if not self.all_preds:
            return {}
        
        # Concatenate all batches
        preds = np.concatenate(self.all_preds)
        targets = np.concatenate(self.all_targets)
        
        metrics = {}
        
        # ====================================================================
        # DIRECTION ACCURACY (Most important for trading!)
        # ====================================================================
        pred_direction = np.sign(preds)
        target_direction = np.sign(targets)
        direction_correct = (pred_direction == target_direction).astype(float)
        metrics['direction_accuracy'] = direction_correct.mean() * 100  # Percentage
        
        # Separate for up/down
        up_mask = target_direction > 0
        down_mask = target_direction < 0
        
        if up_mask.sum() > 0:
            metrics['direction_accuracy_up'] = direction_correct[up_mask].mean() * 100
        if down_mask.sum() > 0:
            metrics['direction_accuracy_down'] = direction_correct[down_mask].mean() * 100
        
        # ====================================================================
        # REGRESSION METRICS
        # ====================================================================
        mae = np.abs(preds - targets).mean()
        mse = ((preds - targets) ** 2).mean()
        rmse = np.sqrt(mse)
        
        metrics['mae'] = mae
        metrics['mse'] = mse
        metrics['rmse'] = rmse
        
        # Relative MAE (as percentage of mean absolute return)
        mean_abs_return = np.abs(targets).mean()
        if mean_abs_return > 0:
            metrics['relative_mae'] = (mae / mean_abs_return) * 100
        
        # ====================================================================
        # CORRELATION
        # ====================================================================
        if len(preds) > 1 and preds.std() > 0 and targets.std() > 0:
            correlation = np.corrcoef(preds, targets)[0, 1]
            metrics['correlation'] = correlation
        else:
            metrics['correlation'] = 0.0
        
        # ====================================================================
        # SHARPE-LIKE METRIC
        # ====================================================================
        # Simulated returns if we trade based on predictions
        simulated_returns = preds * targets  # Profit if direction correct
        if simulated_returns.std() > 0:
            sharpe = simulated_returns.mean() / (simulated_returns.std() + 1e-8)
            metrics['sharpe'] = sharpe
        else:
            metrics['sharpe'] = 0.0
        
        # ====================================================================
        # VOLATILITY METRICS (if available)
        # ====================================================================
        if self.all_volatility_preds:
            vol_preds = np.concatenate(self.all_volatility_preds)
            vol_targets = np.concatenate(self.all_volatility_targets)
            
            vol_mae = np.abs(vol_preds - vol_targets).mean()
            vol_mse = ((vol_preds - vol_targets) ** 2).mean()
            
            metrics['volatility_mae'] = vol_mae
            metrics['volatility_mse'] = vol_mse
            
            if vol_preds.std() > 0 and vol_targets.std() > 0:
                vol_corr = np.corrcoef(vol_preds, vol_targets)[0, 1]
                metrics['volatility_correlation'] = vol_corr
        
        return metrics
    
    def compute_and_reset(self) -> Dict[str, float]:
        """Compute metrics and reset"""
        metrics = self.compute()
        self.reset()
        return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics for printing
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for each line (e.g., "Train: " or "Val: ")
    
    Returns:
        Formatted string
    """
    if not metrics:
        return f"{prefix}No metrics available"
    
    lines = []
    
    # Direction accuracy (most important)
    if 'direction_accuracy' in metrics:
        lines.append(
            f"{prefix}Dir Acc: {metrics['direction_accuracy']:.2f}%"
        )
        if 'direction_accuracy_up' in metrics:
            lines.append(
                f"  ↑ Up: {metrics['direction_accuracy_up']:.2f}%  "
                f"↓ Down: {metrics.get('direction_accuracy_down', 0):.2f}%"
            )
    
    # Error metrics
    if 'mae' in metrics:
        lines.append(
            f"{prefix}MAE: {metrics['mae']:.6f}  "
            f"MSE: {metrics.get('mse', 0):.6f}  "
            f"RMSE: {metrics.get('rmse', 0):.6f}"
        )
    
    # Correlation
    if 'correlation' in metrics:
        lines.append(
            f"{prefix}Correlation: {metrics['correlation']:.4f}  "
            f"Sharpe: {metrics.get('sharpe', 0):.4f}"
        )
    
    # Volatility
    if 'volatility_mae' in metrics:
        lines.append(
            f"{prefix}Vol MAE: {metrics['volatility_mae']:.6f}  "
            f"Vol MSE: {metrics.get('volatility_mse', 0):.6f}"
        )
    
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
