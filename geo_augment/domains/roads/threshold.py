import numpy as np


def threshold_connectivity(field, threshold=0.5):
    return (field >= threshold).astype(np.uint8)
