import numpy as np


def calibrate_flood_risk(
    field: np.ndarray,
    percentile: float,
    value_range: tuple[float, float],
) -> np.ndarray:
    """
    Calibrate flood risk field to target percentile and value range.
    """

    low, high = value_range

    threshold = np.percentile(field, percentile)

    calibrated = field / (threshold + 1e-8)
    calibrated = np.clip(calibrated, low, high)

    return calibrated
