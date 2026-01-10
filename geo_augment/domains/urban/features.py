import numpy as np
from scipy.ndimage import sobel
from geo_augment.io.preprocess import prepare_dem, normalize_minmax


def compute_edge_density(dem):
    dx = sobel(dem, axis=1)
    dy = sobel(dem, axis=0)
    return np.hypot(dx, dy)


def stack_urban_features(dem: np.ndarray) -> np.ndarray:
    dem = prepare_dem(dem)

    edge_density = normalize_minmax(compute_edge_density(dem))
    flatness = 1.0 - edge_density

    return np.stack([dem, edge_density, flatness], axis=0)
