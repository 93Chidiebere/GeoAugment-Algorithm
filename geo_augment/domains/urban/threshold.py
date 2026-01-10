import numpy as np


def threshold_urban_density(field, threshold=0.5):
    return (field >= threshold).astype(np.uint8)
