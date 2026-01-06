import numpy as np


def distribution_summary(
    real: np.ndarray,
    synthetic: np.ndarray
) -> dict:
    """
    Compare basic statistical properties of real vs synthetic flood risk.

    Returns summary metrics useful for sanity checks.
    """

    def stats(arr):
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    return {
        "real": stats(real),
        "synthetic": stats(synthetic),
        "mean_shift": float(np.mean(synthetic) - np.mean(real)),
        "std_ratio": float(
            np.std(synthetic) / (np.std(real) + 1e-8)
        ),
    }


def spatial_correlation(
    real: np.ndarray,
    synthetic: np.ndarray
) -> float:
    """
    Measures spatial agreement using Pearson correlation.

    High correlation indicates structural plausibility.
    """

    r = real.flatten()
    s = synthetic.flatten()

    if r.size != s.size:
        raise ValueError("Arrays must have same shape")

    return float(np.corrcoef(r, s)[0, 1])


def flood_area_ratio(
    binary_real: np.ndarray,
    binary_synthetic: np.ndarray
) -> float:
    """
    Ratio of flooded area (synthetic / real).

    Values close to 1.0 indicate reasonable balance.
    """

    real_area = np.sum(binary_real > 0)
    synth_area = np.sum(binary_synthetic > 0)

    return float(synth_area / (real_area + 1e-8))
