from geo_augment.domains.floods.features import stack_flood_features
from geo_augment.domains.floods.synthesis import generate_synthetic_flood_risk
from geo_augment.domains.floods.threshold import (
    apply_threshold,
    apply_percentile_threshold
)



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

def synthesize_flood_labels(
    dem,
    n_samples: int = 1,
    threshold: float | None = 0.6,
    percentile: float | None = None,
    seed: int | None = None
):
    """
    Generate synthetic binary flood labels.
    """

    features = stack_flood_features(dem)
    slope = features[1]
    base_risk = features[3]

    labels = []

    for i in range(n_samples):
        synthetic_risk = generate_synthetic_flood_risk(
            risk=base_risk,
            slope=slope,
            seed=None if seed is None else seed + i
        )

        if percentile is not None:
            binary = apply_percentile_threshold(
                synthetic_risk,
                percentile=percentile
            )
        else:
            binary = apply_threshold(
                synthetic_risk,
                threshold=threshold
            )

        labels.append(binary)

    return labels
