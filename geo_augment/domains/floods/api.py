import numpy as np

from geo_augment.domains.floods.spec import (
    FloodSynthesisSpec,
    FloodConstraints,
    LatentFloodFieldSpec,
)
from geo_augment.domains.floods.validation import validate_all_flood_specs
from geo_augment.domains.floods.latent import generate_latent_flood_field
from geo_augment.domains.floods.constraints import apply_flood_constraints
from geo_augment.domains.floods.calibration import calibrate_flood_risk


def synthesize_flood_risk(
    dem: np.ndarray,
    synthesis_spec: FloodSynthesisSpec,
    constraints: FloodConstraints,
    latent_spec: LatentFloodFieldSpec,
    n_samples: int = 1,
):
    """
    Generate continuous synthetic flood risk surfaces (0–1).

    This is the canonical GeoAugment flood synthesis API.
    """

    validate_all_flood_specs(
        synthesis=synthesis_spec,
        constraints=constraints,
        latent=latent_spec,
    )

    if n_samples <= 0:
        raise ValueError("n_samples must be >= 1")

    outputs = []

    for i in range(n_samples):
        latent = generate_latent_flood_field(
            dem=dem,
            perturbation_strength=synthesis_spec.perturbation_strength,
            spatial_scale=synthesis_spec.spatial_scale,
            seed=(
                None
                if synthesis_spec.random_seed is None
                else synthesis_spec.random_seed + i
            ),
            latent_spec=latent_spec,
        )

        constrained = apply_flood_constraints(
            latent_field=latent,
            dem=dem,
            constraints=constraints,
        )

        calibrated = calibrate_flood_risk(
            field=constrained,
            percentile=synthesis_spec.risk_percentile,
            value_range=synthesis_spec.value_range,
        )

        outputs.append(calibrated)

    return outputs
