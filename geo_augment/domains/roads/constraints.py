import numpy as np
from scipy.ndimage import gaussian_filter


def apply_road_constraints(
    latent_field,
    features,
    constraints,
):
    dem, gradient, flatness = features

    score = latent_field

    if constraints.enforce_linearity:
        score *= flatness

    if constraints.enforce_smoothness:
        score = gaussian_filter(
            score,
            sigma=constraints.smoothness_kernel_size,
        )

    return np.clip(score, 0.0, 1.0)
