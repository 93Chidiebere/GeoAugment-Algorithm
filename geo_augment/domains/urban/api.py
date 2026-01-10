import numpy as np

from geo_augment.domains.urban.spec import (
    UrbanSynthesisSpec,
    UrbanConstraints,
    LatentUrbanFieldSpec,
)
from geo_augment.domains.urban.validation import validate_urban_specs
from geo_augment.domains.urban.features import stack_urban_features
from geo_augment.domains.urban.synthesis import generate_synthetic_urban_density
from geo_augment.domains.urban.threshold import threshold_urban_density


def synthesize_urban_morphology(
    dem: np.ndarray,
    synthesis_spec: UrbanSynthesisSpec,
    constraints: UrbanConstraints,
    latent_spec: LatentUrbanFieldSpec,
    n_samples: int = 1,
):
    validate_urban_specs(synthesis_spec, constraints, latent_spec)

    features = stack_urban_features(dem)
    outputs = []

    for i in range(n_samples):
        seed = (
            None if synthesis_spec.random_seed is None
            else synthesis_spec.random_seed + i
        )

        outputs.append(
            generate_synthetic_urban_density(
                features,
                synthesis_spec,
                constraints,
                latent_spec,
                seed,
            )
        )

    return outputs


def synthesize_urban_labels(
    density: np.ndarray,
    threshold: float = 0.5,
):
    return threshold_urban_density(density, threshold)
