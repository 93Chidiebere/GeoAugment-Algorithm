import numpy as np


def calibrate_road_connectivity(
    field,
    percentile,
    value_range,
):
    p = np.percentile(field, percentile)
    field = field / (p + 1e-8)
    return np.clip(field, *value_range)
