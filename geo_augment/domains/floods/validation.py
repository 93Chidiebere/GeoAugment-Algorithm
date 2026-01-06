from typing import Tuple

from geo_augment.domains.floods.spec import (
    FloodSynthesisSpec,
    FloodConstraints,
    LatentFloodFieldSpec,
)


class FloodSpecValidationError(ValueError):
    """Raised when flood synthesis parameters are invalid."""


def _validate_range(
    name: str,
    value: float,
    allowed: Tuple[float, float],
):
    low, high = allowed
    if not (low <= value <= high):
        raise FloodSpecValidationError(
            f"{name}={value} is outside allowed range [{low}, {high}]"
        )


def validate_flood_synthesis_spec(spec: FloodSynthesisSpec) -> None:
    """Validate core flood synthesis parameters."""

    _validate_range(
        "perturbation_strength",
        spec.perturbation_strength,
        (0.0, 1.0),
    )

    if spec.spatial_scale <= 0:
        raise FloodSpecValidationError(
            "spatial_scale must be > 0"
        )

    _validate_range(
        "risk_percentile",
        spec.risk_percentile,
        (0.0, 100.0),
    )

    vmin, vmax = spec.value_range
    if vmin >= vmax:
        raise FloodSpecValidationError(
            f"value_range must be (min, max) with min < max, got {spec.value_range}"
        )


def validate_flood_constraints(constraints: FloodConstraints) -> None:
    """Validate physical and statistical constraints."""

    if constraints.smoothness_kernel_size <= 0:
        raise FloodSpecValidationError(
            "smoothness_kernel_size must be positive"
        )

    if constraints.smoothness_kernel_size % 2 == 0:
        raise FloodSpecValidationError(
            "smoothness_kernel_size must be an odd integer"
        )

    if constraints.downhill_weight < 0:
        raise FloodSpecValidationError(
            "downhill_weight must be >= 0"
        )

    if (
        constraints.enforce_spatial_smoothness
        and constraints.smoothness_kernel_size < 3
    ):
        raise FloodSpecValidationError(
            "smoothness_kernel_size must be >= 3 when spatial smoothness is enforced"
        )


def validate_latent_field_spec(spec: LatentFloodFieldSpec) -> None:
    """Validate latent flood field configuration."""

    allowed_noise_types = {"gaussian", "perlin", "spectral"}

    if spec.noise_type not in allowed_noise_types:
        raise FloodSpecValidationError(
            f"noise_type='{spec.noise_type}' is invalid. "
            f"Allowed values: {sorted(allowed_noise_types)}"
        )


def validate_all_flood_specs(
    synthesis: FloodSynthesisSpec,
    constraints: FloodConstraints,
    latent: LatentFloodFieldSpec,
) -> None:
    """Convenience validator for full flood synthesis stack."""
    validate_flood_synthesis_spec(synthesis)
    validate_flood_constraints(constraints)
    validate_latent_field_spec(latent)
