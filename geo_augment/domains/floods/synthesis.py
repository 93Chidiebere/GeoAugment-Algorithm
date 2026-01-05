import numpy as np
from scipy.ndimage import gaussian_filter


def generate_spatial_noise(
    shape: tuple,
    scale: float = 0.05,
    smooth_sigma: float = 3.0,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate smooth spatial noise.
    """
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.normal(0.0, scale, size=shape)
    return gaussian_filter(noise, sigma=smooth_sigma)


# Hydrology-aware perturbation
def perturb_flood_risk(
    risk: np.ndarray,
    slope: np.ndarray,
    max_delta: float = 0.15,
    smooth_sigma: float = 2.0,
    seed: int | None = None
) -> np.ndarray:
    """
    Apply controlled perturbation to flood risk.
    """

    noise = generate_spatial_noise(
        shape=risk.shape,
        scale=max_delta,
        smooth_sigma=smooth_sigma,
        seed=seed
    )

    # Penalize perturbation on steep slopes
    damping = 1.0 - slope
    perturbed = risk + noise * damping

    return np.clip(perturbed, 0.0, 1.0)


# Constraint validation
def validate_flood_risk(
    original: np.ndarray,
    synthetic: np.ndarray,
    max_mean_shift: float = 0.1,
    max_pixel_change: float = 0.3
) -> bool:
    """
    Validate synthetic risk plausibility.
    """

    mean_shift = np.abs(synthetic.mean() - original.mean())
    max_change = np.abs(synthetic - original).max()

    if mean_shift > max_mean_shift:
        return False

    if max_change > max_pixel_change:
        return False

    return True


# Rejection Sampling wrapper
def generate_synthetic_flood_risk(
    risk: np.ndarray,
    slope: np.ndarray,
    attempts: int = 10,
    **kwargs
) -> np.ndarray:
    """
    Generate one valid synthetic flood risk sample.
    """

    for _ in range(attempts):
        candidate = perturb_flood_risk(
            risk=risk,
            slope=slope,
            **kwargs
        )

        if validate_flood_risk(risk, candidate):
            return candidate

    raise RuntimeError("Failed to generate valid synthetic flood risk.")
