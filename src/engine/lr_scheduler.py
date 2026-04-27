"""
Learning Rate Schedulers for CARD Training
Implements linear warmup with cosine annealing or step decay.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    """
    LR scheduler with linear warmup followed by cosine annealing

    Schedule:
        Epochs 0-warmup_epochs: Linear increase from 0 to base_lr
        Epochs warmup_epochs-max_epochs: Cosine decay to min_lr

    This is the RECOMMENDED scheduler for Phase 4.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs (default 3)
        max_epochs: Total number of training epochs (default 200)
        base_lr: Peak learning rate after warmup (default 1e-4)
        min_lr: Minimum learning rate (default 1e-6)
        last_epoch: Last epoch number (for resuming)

    Example:
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler = WarmupCosineLR(optimizer, warmup_epochs=3, max_epochs=200)

        for epoch in range(200):
            train(...)
            scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs=3,
        max_epochs=200,
        base_lr=1e-4,
        min_lr=1e-6,
        last_epoch=-1,
    ):

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current epoch"""

        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Warmup phase: linear increase from 0 to base_lr
            if self.warmup_epochs == 0:
                lr = self.base_lr
            else:
                lr = self.base_lr * (epoch / self.warmup_epochs)
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            progress = min(progress, 1.0)  # Clip to [0, 1]

            # Cosine formula: lr = min_lr + (base_lr - min_lr) × 0.5 × (1 + cos(π × progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        return [lr for _ in self.optimizer.param_groups]


class WarmupStepLR(_LRScheduler):
    """
    LR scheduler with linear warmup followed by step decay

    Alternative to cosine schedule (less common, but available)

    Schedule:
        Epochs 0-warmup_epochs: Linear increase from 0 to base_lr
        Epochs warmup_epochs+: Step decay every step_size epochs

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        step_size: Decay LR every N epochs after warmup
        gamma: Multiplicative decay factor (default 0.1)
        base_lr: Peak learning rate
        last_epoch: Last epoch number
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs=3,
        step_size=30,
        gamma=0.1,
        base_lr=1e-4,
        last_epoch=-1,
    ):

        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = base_lr

        super(WarmupStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current epoch"""

        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Warmup phase
            if self.warmup_epochs == 0:
                lr = self.base_lr
            else:
                lr = self.base_lr * (epoch / self.warmup_epochs)
        else:
            # Step decay phase
            decay_steps = (epoch - self.warmup_epochs) // self.step_size
            lr = self.base_lr * (self.gamma**decay_steps)

        return [lr for _ in self.optimizer.param_groups]


def visualize_schedule(scheduler_class, **kwargs):
    """
    Utility to visualize LR schedule

    Example:
        visualize_schedule(WarmupCosineLR, warmup_epochs=3, max_epochs=200)
    """

    import matplotlib.pyplot as plt

    # Dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=kwargs.get("base_lr", 1e-4))

    # Create scheduler
    scheduler = scheduler_class(optimizer, **kwargs)

    # Simulate epochs
    max_epochs = kwargs.get("max_epochs", 200)
    lrs = []

    for epoch in range(max_epochs):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(f"{scheduler_class.__name__} Schedule")
    plt.grid(True)
    plt.yscale("log")
    plt.show()

    # Print key values
    print(f"Epoch 0 LR: {lrs[0]:.2e}")
    print(
        f"Epoch {kwargs.get('warmup_epochs', 3)} LR: {lrs[kwargs.get('warmup_epochs', 3)]:.2e}"
    )
    print(f"Epoch {max_epochs // 2} LR: {lrs[max_epochs // 2]:.2e}")
    print(f"Epoch {max_epochs - 1} LR: {lrs[-1]:.2e}")
