import numpy as np
from geo_augment.domains.floods.flow_direction import D8_OFFSETS


def compute_flow_accumulation(flow_dir: np.ndarray) -> np.ndarray:
    """
    Compute flow accumulation using recursive upstream counting.
    
    Returns:
        acc: float array of shape (H, W)
    """
    H, W = flow_dir.shape
    acc = np.ones((H, W), dtype=np.float32)

    # Build reverse flow graph
    upstream = [[[] for _ in range(W)] for _ in range(H)]

    for i in range(H):
        for j in range(W):
            d = flow_dir[i, j]
            if d >= 0:
                di, dj = D8_OFFSETS[d]
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    upstream[ni][nj].append((i, j))

    visited = np.zeros((H, W), dtype=bool)

    def dfs(i, j):
        if visited[i, j]:
            return acc[i, j]
        visited[i, j] = True

        for ui, uj in upstream[i][j]:
            acc[i, j] += dfs(ui, uj)

        return acc[i, j]

    for i in range(H):
        for j in range(W):
            if not visited[i, j]:
                dfs(i, j)

    return acc
