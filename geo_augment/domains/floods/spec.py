from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class FloodSynthesisSpec:
    """
    Formal specification for flood risk synthesis.

    This defines WHAT is allowed to vary and WHAT must be preserved.
    """

    # -----------------------------
    # Latent flood driver parameters
    # -----------------------------

    perturbation_strength: float
    """
    Controls magnitude of latent flood potential variation.
    Typical range: [0.05, 0.3]
    """

    spatial_scale: float
    """
    Characteristic length scale (in pixels) of perturbations.
    Larger values = smoother flood patterns.
    """

    random_seed: Optional[int] = None
    """
    Ensures reproducibility when set.
    """

    # -----------------------------
    # Calibration parameters
    # -----------------------------

    risk_percentile: float = 90.0
    """
    Percentile used to normalize flood risk.
    Interpreted as 'high-risk threshold'.
    """

    # -----------------------------
    # Output constraints
    # -----------------------------

    value_range: Tuple[float, float] = (0.0, 1.0)
    """
    Enforced bounds on output flood risk.
    """

    enforce_smoothness: bool = True
    """
    Enforce spatial smoothness constraint.
    """

    enforce_downhill_bias: bool = True
    """
    Bias flood risk to increase in lower elevations.
    """

# Constraint definitions
@dataclass(frozen=True)
class FloodConstraints:
    """
    Physical and statistical constraints applied to flood synthesis.
    """

    enforce_bounds: bool = True
    enforce_monotonic_downhill: bool = True
    enforce_spatial_smoothness: bool = True

    smoothness_kernel_size: int = 5
    """
    Size of spatial smoothing kernel (odd integer).
    """

    downhill_weight: float = 1.0
    """
    Strength of downhill bias relative to other factors.
    """

# Latent field configuration
@dataclass(frozen=True)
class LatentFloodFieldSpec:
    """
    Controls how latent flood potential is generated.
    """

    noise_type: str = "gaussian"
    """
    Type of base noise: 'gaussian' | 'perlin' | 'spectral'
    """

    normalize: bool = True
    """
    Whether to normalize latent field before constraints.
    """

    apply_low_frequency_bias: bool = True
    """
    Enforce dominance of low-frequency spatial patterns.

    """

DEFAULT_FLOOD_SPEC = FloodSynthesisSpec(
    perturbation_strength=0.15,
    spatial_scale=30.0,
    random_seed=None,
    risk_percentile=90.0
)

DEFAULT_FLOOD_CONSTRAINTS = FloodConstraints()

DEFAULT_LATENT_SPEC = LatentFloodFieldSpec()
