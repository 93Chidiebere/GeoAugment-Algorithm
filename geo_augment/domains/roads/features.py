import numpy as np
from scipy.ndimage import sobel
from geo_augment.io.preprocess import prepare_dem, normalize_minmax


def compute_gradient_magnitude(dem):
    dx = sobel(dem, axis=1)
    dy = sobel(dem, axis=0)
    return np.hypot(dx, dy)


def stack_road_features(dem: np.ndarray) -> np.ndarray:
    dem = prepare_dem(dem)

    gradient = normalize_minmax(compute_gradient_magnitude(dem))
    flatness = 1.0 - gradient

    return np.stack([dem, gradient, flatness], axis=0)
