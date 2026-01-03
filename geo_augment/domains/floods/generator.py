# Flood synthetic generator

import numpy as np
from geo_augment.core.sampler import LatentSampler
from geo_augment.domains.floods.constraints import FloodConstraints


class FloodSyntheticGenerator:
    """
    End-to-end flood synthetic data generator.
    """

    def __init__(
        self,
        sampler: LatentSampler = None,
        constraints: FloodConstraints = None
    ):
        self.sampler = sampler or LatentSampler()
        self.constraints = constraints or FloodConstraints()

    def generate(
        self,
        features: np.ndarray
    ):
        """
        Generates synthetic features and continuous flood risk labels.
        """
        synthetic_features = self.sampler.perturb(features)

        elevation = synthetic_features[0]
        slope = synthetic_features[1]

        risk = self.constraints.risk_score(elevation, slope)

        return {
            "features": synthetic_features,
            "risk": risk,
            "synthetic_mask": np.ones_like(risk, dtype=bool)
        }
