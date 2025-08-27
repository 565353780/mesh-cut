import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1)
    normed_vectors = np.zeros_like(vectors)

    valid_norm_idxs = np.where(norm > 0)[0]

    normed_vectors[valid_norm_idxs] = vectors[valid_norm_idxs] / norm[
        valid_norm_idxs
    ].reshape(-1, 1)

    return normed_vectors
