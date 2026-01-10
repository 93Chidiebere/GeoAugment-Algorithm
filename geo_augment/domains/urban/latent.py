import numpy as np
from scipy.ndimage import gaussian_filter


def generate_latent_urban_field(
    shape,
    perturbation_strength,
    spatial_scale,
    seed,
    latent_spec,
):
    rng = np.random.default_rng(seed)

    noise = rng.standard_normal(shape)
    field = gaussian_filter(noise, sigma=spatial_scale)

    field *= perturbation_strength

    if latent_spec.normalize:
        field = (field - field.min()) / (field.max() - field.min())

    return field
