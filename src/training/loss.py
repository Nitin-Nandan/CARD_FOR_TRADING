import torch
import torch.nn as nn
import math


class CombinedReturnLoss(nn.Module):
    """
    Combined Loss: Signal Decay MAE + Directional Penalty.
    """

    def __init__(self, directional_weight=10.0, magnitude_scale=1000.0):
        super().__init__()
        self.directional_weight = directional_weight
        self.magnitude_scale = magnitude_scale

    def forward(self, pred_returns, true_returns):
        """
        Args:
            pred_returns: (batch, 15) predicted returns
            true_returns: (batch, 15) true returns
        """
        batch_size, pred_len = pred_returns.shape

        # 1. MAGNITUDE LOSS (Signal Decay MAE)
        mae = torch.abs(pred_returns - true_returns)

        # Time decay weights (CARD Eq. 12)
        decay_weights = torch.tensor(
            [1.0 / math.sqrt(i + 1) for i in range(pred_len)],
            device=pred_returns.device,
        )

        weighted_mae = (mae * decay_weights).mean()
        scaled_mae = weighted_mae * self.magnitude_scale

        # 2. DIRECTIONAL LOSS
        pred_sign = torch.sign(pred_returns)
        true_sign = torch.sign(true_returns)

        correct_direction = (pred_sign == true_sign).float()
        direction_accuracy = correct_direction.mean().item()

        direction_penalty = 1.0 - correct_direction.mean()
        scaled_direction_penalty = direction_penalty * self.directional_weight

        # 3. COMBINED LOSS
        total_loss = scaled_mae + scaled_direction_penalty

        # 4. METRICS
        loss_dict = {
            "total": total_loss.item(),
            "weighted_mae": weighted_mae.item(),
            "scaled_mae": scaled_mae.item(),
            "direction_penalty": direction_penalty.item(),
            "scaled_direction_penalty": scaled_direction_penalty.item(),
            "direction_accuracy": direction_accuracy,
        }

        return total_loss, loss_dict
