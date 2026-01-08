import numpy as np
from scipy.ndimage import sobel

from geo_augment.io.preprocess import prepare_dem, normalize_minmax
from geo_augment.domains.floods.flow_direction import compute_flow_direction
from geo_augment.domains.floods.flow_accumulation import compute_flow_accumulation
from geo_augment.domains.floods.risk import compute_flood_risk


def compute_slope(dem: np.ndarray) -> np.ndarray:
    """
    Compute terrain slope magnitude using Sobel gradients.
    """
    dx = sobel(dem, axis=1)
    dy = sobel(dem, axis=0)
    slope = np.hypot(dx, dy)
    return slope


def stack_flood_features(dem: np.ndarray) -> np.ndarray:
    """
    Stack flood-relevant terrain features.

    Parameters
    ----------
    dem : np.ndarray
        Raw digital elevation model.

    Returns
    -------
    np.ndarray
        Shape: (4, H, W)

        Channels:
        0 - Elevation (normalized)
        1 - Slope (normalized)
        2 - Flow accumulation (normalized)
        3 - Base flood risk proxy (continuous, 0–1)

    Notes
    -----
    These features are:
    - Optional for GeoAugment synthesis
    - Intended for exploration and ML model inputs
    - Not flood predictions
    """

    # -----------------------------
    # 1. Preprocess DEM
    # -----------------------------
    dem = prepare_dem(dem)
    elevation = normalize_minmax(dem)

    # -----------------------------
    # 2. Slope
    # -----------------------------
    slope_raw = compute_slope(elevation)
    slope = normalize_minmax(slope_raw)

    # -----------------------------
    # 3. Flow accumulation
    # -----------------------------
    flow_dir = compute_flow_direction(elevation)
    flow_acc_raw = compute_flow_accumulation(flow_dir)
    flow_acc = normalize_minmax(flow_acc_raw)

    # -----------------------------
    # 4. Base flood risk proxy
    # -----------------------------
    risk = compute_flood_risk(
        elevation=elevation,
        slope=slope,
        flow_acc=flow_acc,
    )

    return np.stack(
        [elevation, slope, flow_acc, risk],
        axis=0,
    )
