import numpy as np

from geo_augment.domains.roads.spec import (
    RoadSynthesisSpec,
    RoadConstraints,
    LatentRoadFieldSpec,
)
from geo_augment.domains.roads.validation import validate_road_specs
from geo_augment.domains.roads.features import stack_road_features
from geo_augment.domains.roads.synthesis import generate_synthetic_road_connectivity
from geo_augment.domains.roads.threshold import threshold_connectivity


def synthesize_road_connectivity(
    dem: np.ndarray,
    synthesis_spec: RoadSynthesisSpec,
    constraints: RoadConstraints,
    latent_spec: LatentRoadFieldSpec,
    n_samples: int = 1,
):
    validate_road_specs(synthesis_spec, constraints, latent_spec)

    features = stack_road_features(dem)
    outputs = []

    for i in range(n_samples):
        seed = (
            None if synthesis_spec.random_seed is None
            else synthesis_spec.random_seed + i
        )

        outputs.append(
            generate_synthetic_road_connectivity(
                features,
                synthesis_spec,
                constraints,
                latent_spec,
                seed,
            )
        )

    return outputs


def synthesize_road_labels(
    connectivity: np.ndarray,
    threshold: float = 0.5,
):
    return threshold_connectivity(connectivity, threshold)
