import numpy as np
from tqdm import trange


def toFPSIdxs(points: np.ndarray, sample_point_num: int) -> np.ndarray:
    N = points.shape[0]
    sampled_indices = np.zeros(sample_point_num, dtype=int)
    distances = np.full(N, np.inf)

    # 初始化：从一个随机点开始
    sampled_indices[0] = np.random.randint(N)
    farthest = points[sampled_indices[0]]

    print("[INFO][sample::toFPSIdxs]")
    print("\t start sample fps points...")
    for i in trange(1, sample_point_num):
        dist = np.linalg.norm(points - farthest, axis=1)
        distances = np.minimum(distances, dist)
        sampled_indices[i] = np.argmax(distances)
        farthest = points[sampled_indices[i]]

    return sampled_indices
