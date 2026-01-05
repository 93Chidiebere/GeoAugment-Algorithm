import os
import numpy as np
from typing import Optional, Dict


def export_npz(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    name: str = "geoaugment_flood",
    metadata: Optional[Dict] = None
):
    """
    Export dataset to NumPy .npz format.
    """

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{name}.npz")

    if metadata is None:
        metadata = {}

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        **metadata
    )


# PyTorch Export
def export_torch(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    name: str = "geoaugment_flood"
):
    """
    Export dataset to PyTorch .pt format.

    Requires torch to be installed.
    """

    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is not installed. "
            "Install it to use export_torch()."
        )

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{name}.pt")

    dataset = {
        "X": torch.from_numpy(X).float(),
        "y": torch.from_numpy(y).long()
    }

    torch.save(dataset, out_path)
