import numpy as np


def apply_threshold(
    risk: np.ndarray,
    threshold: float = 0.6
) -> np.ndarray:
    """
    Convert continuous flood risk to binary flood labels.

    Parameters
    ----------
    risk : np.ndarray
        Flood risk surface in [0, 1]
    threshold : float
        Decision threshold

    Returns
    -------
    binary : np.ndarray
        Binary flood mask (0 or 1)
    """
    return (risk >= threshold).astype(np.uint8)

# Percentile-based thresholding - data-scarce regions
def apply_percentile_threshold(
    risk: np.ndarray,
    percentile: float = 90.0
) -> np.ndarray:
    """
    Threshold flood risk using percentile-based cutoff.

    Parameters
    ----------
    risk : np.ndarray
        Flood risk surface
    percentile : float
        Percentile (0–100)

    Returns
    -------
    binary : np.ndarray
        Binary flood mask
    """
    cutoff = np.percentile(risk, percentile)
    return (risk >= cutoff).astype(np.uint8)
