from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class UrbanSynthesisSpec:
    perturbation_strength: float
    spatial_scale: float
    random_seed: Optional[int] = None

    density_percentile: float = 80.0
    value_range: Tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class UrbanConstraints:
    enforce_compactness: bool = True
    enforce_spatial_smoothness: bool = True
    enforce_density_bias: bool = True

    smoothness_kernel_size: int = 7
    density_weight: float = 1.0


@dataclass(frozen=True)
class LatentUrbanFieldSpec:
    noise_type: str = "spectral"
    normalize: bool = True
    apply_grid_bias: bool = True


DEFAULT_URBAN_SPEC = UrbanSynthesisSpec(
    perturbation_strength=0.18,
    spatial_scale=25.0,
)

DEFAULT_URBAN_CONSTRAINTS = UrbanConstraints()
DEFAULT_LATENT_URBAN_SPEC = LatentUrbanFieldSpec()
