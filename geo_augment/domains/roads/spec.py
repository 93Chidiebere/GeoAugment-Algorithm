from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class RoadSynthesisSpec:
    perturbation_strength: float
    spatial_scale: float
    random_seed: Optional[int] = None

    connectivity_percentile: float = 85.0
    value_range: Tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class RoadConstraints:
    enforce_smoothness: bool = True
    enforce_linearity: bool = True
    enforce_network_continuity: bool = True

    smoothness_kernel_size: int = 5
    continuity_weight: float = 1.0


@dataclass(frozen=True)
class LatentRoadFieldSpec:
    noise_type: str = "spectral"
    normalize: bool = True
    apply_directional_bias: bool = True


DEFAULT_ROAD_SPEC = RoadSynthesisSpec(
    perturbation_strength=0.12,
    spatial_scale=20.0,
)

DEFAULT_ROAD_CONSTRAINTS = RoadConstraints()
DEFAULT_LATENT_ROAD_SPEC = LatentRoadFieldSpec()
