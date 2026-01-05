from geo_augment.domains.floods.features import stack_flood_features
from geo_augment.domains.floods.synthesis import generate_synthetic_flood_risk


def synthesize_flood_risk(
    dem,
    n_samples: int = 1,
    seed: int | None = None
):
    """
    Generate synthetic flood risk samples from DEM.
    """

    features = stack_flood_features(dem)
    slope = features[1]
    base_risk = features[3]

    samples = []

    for i in range(n_samples):
        sample = generate_synthetic_flood_risk(
            risk=base_risk,
            slope=slope,
            seed=None if seed is None else seed + i
        )
        samples.append(sample)

    return samples
