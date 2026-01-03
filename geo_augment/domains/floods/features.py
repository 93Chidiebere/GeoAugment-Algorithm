import numpy as np
from scipy.ndimage import sobel
from geo_augment.io.preprocess import prepare_dem, normalize_minmax


def compute_slope(dem: np.ndarray) -> np.ndarray:
    """
    Approximate slope using Sobel gradients.
    """
    dx = sobel(dem, axis=1)
    dy = sobel(dem, axis=0)
    slope = np.hypot(dx, dy)
    return slope


def stack_flood_features(dem: np.ndarray) -> np.ndarray:
    """
    Create flood feature stack:
    Channel 0: elevation
    Channel 1: slope
    """
    dem = prepare_dem(dem)
    slope = compute_slope(dem)
    slope = normalize_minmax(slope)

    features = np.stack([dem, slope], axis=0)
    return features
