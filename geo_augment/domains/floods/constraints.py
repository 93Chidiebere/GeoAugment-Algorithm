import numpy as np


class FloodConstraints:
    """
    Physics-inspired constraints for flood susceptibility.
    Enforces monotonic relationships without hard simulation.
    """

    def __init__(
        self,
        elevation_weight: float = 0.6,
        slope_weight: float = 0.4
    ):
        self.elevation_weight = elevation_weight
        self.slope_weight = slope_weight

    def risk_score(
        self,
        elevation: np.ndarray,
        slope: np.ndarray
    ) -> np.ndarray:
        """
        Computes continuous flood risk score (0–1).
        Lower elevation and lower slope increase risk.
        """
        elevation_term = 1.0 - elevation
        slope_term = 1.0 - slope

        risk = (
            self.elevation_weight * elevation_term +
            self.slope_weight * slope_term
        )

        return np.clip(risk, 0.0, 1.0)
    