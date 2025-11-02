import numpy as np


def nearest_front_index(path, xy, heading_vec):
    rel = path - xy[None, :]
    dot = rel @ heading_vec
    mask = dot >= 0.0
    d2 = np.sum(rel[mask] ** 2, axis=1)
    if not np.any(mask):
        return int(path.shape[0])
    idx_in_mask = int(np.argmin(d2))
    return int(np.arange(len(path))[mask][idx_in_mask])

