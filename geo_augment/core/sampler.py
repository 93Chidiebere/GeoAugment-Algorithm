import numpy as np


class LatentSampler:
    """
    Generates constrained perturbations in feature space.
    """

    def __init__(self, noise_scale: float = 0.05):
        self.noise_scale = noise_scale

    def perturb(self, features: np.ndarray) -> np.ndarray:
        """
        Adds controlled Gaussian noise to features.
        """
        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_scale,
            size=features.shape
        )
        synthetic = features + noise
        return np.clip(synthetic, 0.0, 1.0)
