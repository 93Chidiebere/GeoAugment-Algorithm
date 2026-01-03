import numpy as np


# Clockwise from North
D8_OFFSETS = np.array([
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1)   # NW
])


def compute_flow_direction(dem: np.ndarray) -> np.ndarray:
    """
    Compute D8 flow direction.
    
    Returns:
        flow_dir: int array of shape (H, W)
                  values in [0..7], -1 for sinks
    """
    H, W = dem.shape
    flow_dir = -1 * np.ones((H, W), dtype=np.int8)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            center = dem[i, j]
            drops = []

            for k, (di, dj) in enumerate(D8_OFFSETS):
                ni, nj = i + di, j + dj
                drops.append(center - dem[ni, nj])

            drops = np.array(drops)
            max_drop = drops.max()

            if max_drop > 0:
                flow_dir[i, j] = int(drops.argmax())

    return flow_dir
