import numpy as np
from scipy.ndimage import gaussian_filter


def apply_flood_constraints(
    latent_field: np.ndarray,
    dem: np.ndarray,
    constraints,
) -> np.ndarray:
    """
    Apply physical and statistical constraints to latent flood field.
    """

    field = latent_field.copy()

    if constraints.enforce_monotonic_downhill:
        elevation_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
        downhill_bias = 1.0 - elevation_norm
        field = field + constraints.downhill_weight * downhill_bias

    if constraints.enforce_spatial_smoothness:
        field = gaussian_filter(field, sigma=constraints.smoothness_kernel_size)

    if constraints.enforce_bounds:
        field = np.clip(field, 0.0, 1.0)

    return field

    