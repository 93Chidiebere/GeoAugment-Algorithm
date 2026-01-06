import numpy as np

from geo_augment.domains.floods.spec import (
    FloodSynthesisSpec,
    FloodConstraints,
    LatentFloodFieldSpec,
)
from geo_augment.domains.floods.validation import (
    validate_all_flood_specs,
)
from geo_augment.domains.floods.latent import generate_latent_flood_field
from geo_augment.domains.floods.constraints import apply_flood_constraints
from geo_augment.domains.floods.calibration import calibrate_flood_risk


def synthesize_flood_labels(
    dem: np.ndarray,
    synthesis_spec: FloodSynthesisSpec,
    constraints: FloodConstraints,
    latent_spec: LatentFloodFieldSpec,
    n_samples: int = 1,
):
    """
    Generate synthetic flood risk labels from DEM.

    Validation is enforced before any computation.
    """

    # -----------------------------
    # 1. Validate all specs (fail fast)
    # -----------------------------
    validate_all_flood_specs(
        synthesis=synthesis_spec,
        constraints=constraints,
        latent=latent_spec,
    )

    if n_samples <= 0:
        raise ValueError("n_samples must be >= 1")

    outputs = []

    for i in range(n_samples):
        # -----------------------------
        # 2. Generate latent flood field
        # -----------------------------
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

        # -----------------------------
        # 3. Apply physical constraints
        # -----------------------------
        constrained = apply_flood_constraints(
            latent_field=latent,
            dem=dem,
            constraints=constraints,
        )

        # -----------------------------
        # 4. Calibrate to risk percentile
        # -----------------------------
        calibrated = calibrate_flood_risk(
            field=constrained,
            percentile=synthesis_spec.risk_percentile,
            value_range=synthesis_spec.value_range,
        )

        outputs.append(calibrated)

    return outputs
