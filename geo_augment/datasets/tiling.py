import numpy as np
from typing import Tuple, List


def tile_raster(
    features: np.ndarray,
    labels: np.ndarray,
    tile_size: int = 256,
    overlap: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tile feature and label rasters into fixed-size patches.

    Parameters
    ----------
    features : np.ndarray
        Feature tensor of shape (C, H, W)
    labels : np.ndarray
        Label raster of shape (H, W)
    tile_size : int
        Tile height and width
    overlap : int
        Number of overlapping pixels between tiles

    Returns
    -------
    X : np.ndarray
        Feature tiles of shape (N, C, tile_size, tile_size)
    y : np.ndarray
        Label tiles of shape (N, tile_size, tile_size)
    """

    assert features.ndim == 3, "Features must be (C, H, W)"
    assert labels.ndim == 2, "Labels must be (H, W)"

    C, H, W = features.shape
    stride = tile_size - overlap

    X_tiles: List[np.ndarray] = []
    y_tiles: List[np.ndarray] = []

    for row in range(0, H - tile_size + 1, stride):
        for col in range(0, W - tile_size + 1, stride):

            feat_tile = features[
                :,
                row:row + tile_size,
                col:col + tile_size
            ]

            label_tile = labels[
                row:row + tile_size,
                col:col + tile_size
            ]

            if feat_tile.shape[1:] != (tile_size, tile_size):
                continue

            X_tiles.append(feat_tile)
            y_tiles.append(label_tile)

    X = np.stack(X_tiles)
    y = np.stack(y_tiles)

    return X, y
