import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_flood_risk(
    elevation: np.ndarray,
    slope: np.ndarray,
    flow_acc: np.ndarray,
    weights: dict | None = None
) -> np.ndarray:
    """
    Compute continuous flood risk surface in [0, 1].

    Parameters
    ----------
    elevation : np.ndarray
        Normalized elevation [0, 1]
    slope : np.ndarray
        Normalized slope [0, 1]
    flow_acc : np.ndarray
        Normalized flow accumulation [0, 1]
    weights : dict
        Optional weights: {'elev', 'slope', 'acc'}

    Returns
    -------
    risk : np.ndarray
        Flood risk surface [0, 1]
    """

    if weights is None:
        weights = {
            "elev": 1.0,
            "slope": 1.0,
            "acc": 1.5
        }

    score = (
        weights["elev"] * (1.0 - elevation) +
        weights["slope"] * (1.0 - slope) +
        weights["acc"] * flow_acc
    )

    return sigmoid(score)
