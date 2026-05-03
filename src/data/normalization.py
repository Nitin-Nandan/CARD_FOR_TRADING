import numpy as np

def normalize_features(X):
    """Standard implementation of normalized safety clipping."""
    # 1. Wide clipping to prevent physical saturation of float16/float32
    X_clipped = np.clip(X, -1000000.0, 1000000.0)

    # 2. Handle any NaNs or Infs
    X_clean = np.nan_to_num(X_clipped, nan=0.0, posinf=0.0, neginf=0.0)

    return X_clean
