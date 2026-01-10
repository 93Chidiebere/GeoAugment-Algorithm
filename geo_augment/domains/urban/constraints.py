import numpy as np
from scipy.ndimage import gaussian_filter


def apply_urban_constraints(
    latent_field,
    features,
    constraints,
):
    dem, edge_density, flatness = features

    score = latent_field

    if constraints.enforce_density_bias:
        score *= flatness

    if constraints.enforce_compactness:
        score = gaussian_filter(
            score,
            sigma=constraints.smoothness_kernel_size,
        )

    return np.clip(score, 0.0, 1.0)
