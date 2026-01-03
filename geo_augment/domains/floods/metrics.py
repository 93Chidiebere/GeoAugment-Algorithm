import numpy as np


def risk_to_binary(
    risk: np.ndarray,
    threshold: float = 0.6
) -> np.ndarray:
    """
    Converts continuous flood risk to binary mask.
    """
    return (risk >= threshold).astype(int)
