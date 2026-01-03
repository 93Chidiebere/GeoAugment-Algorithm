import numpy as np
from geo_augment.io.preprocess import prepare_dem, normalize_minmax
from geo_augment.domains.floods.flow_direction import compute_flow_direction
from geo_augment.domains.floods.flow_accumulation import compute_flow_accumulation
from geo_augment.domains.floods.risk import compute_flood_risk

from scipy.ndimage import sobel


def compute_slope(dem: np.ndarray) -> np.ndarray:
    dx = sobel(dem, axis=1)
    dy = sobel(dem, axis=0)
    return np.hypot(dx, dy)


def stack_flood_features(dem: np.ndarray) -> np.ndarray:
    """
    Channels:
    0: elevation
    1: slope
    2: flow accumulation
    3: flood risk (continuous)
    """
    dem = prepare_dem(dem)

    slope = normalize_minmax(compute_slope(dem))

    flow_dir = compute_flow_direction(dem)
    flow_acc = normalize_minmax(compute_flow_accumulation(flow_dir))

    risk = compute_flood_risk(
        elevation=dem,
        slope=slope,
        flow_acc=flow_acc
    )

    return np.stack([dem, slope, flow_acc, risk], axis=0)

