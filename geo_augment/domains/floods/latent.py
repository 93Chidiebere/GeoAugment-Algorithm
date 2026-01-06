import numpy as np
from scipy.ndimage import gaussian_filter


def generate_latent_flood_field(
    dem: np.ndarray,
    perturbation_strength: float,
    spatial_scale: float,
    seed: int | None,
    latent_spec,
) -> np.ndarray:
    """
    Generate a latent flood potential field.
    """

    if seed is not None:
        np.random.seed(seed)

    noise = np.random.normal(
        loc=0.0,
        scale=perturbation_strength,
        size=dem.shape,
    )

    smooth_noise = gaussian_filter(noise, sigma=spatial_scale)

    latent = smooth_noise

    if latent_spec.normalize:
        latent = (latent - latent.min()) / (latent.max() - latent.min() + 1e-8)

    return latent
