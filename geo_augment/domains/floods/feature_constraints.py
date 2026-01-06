# geo_augment/domains/floods/feature_constraints.py

import numpy as np


class FeatureFloodConstraints:
    """
    Feature-level constraints for flood susceptibility.
    Combines elevation and slope into continuous risk.
    """

    def __init__(
        self,
        elevation_weight: float = 0.6,
        slope_weight: float = 0.4,
    ):
        self.elevation_weight = elevation_weight
        self.slope_weight = slope_weight

    def risk_score(
        self,
        elevation: np.ndarray,
        slope: np.ndarray,
    ) -> np.ndarray:
        elevation_term = 1.0 - elevation
        slope_term = 1.0 - slope

        risk = (
            self.elevation_weight * elevation_term
            + self.slope_weight * slope_term
        )

        return np.clip(risk, 0.0, 1.0)
