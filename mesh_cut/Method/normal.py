import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)

        if norm == 0:
            return vectors

        return vectors / norm

    if vectors.ndim == 2:
        norm = np.linalg.norm(vectors, axis=1)
        normed_vectors = np.zeros_like(vectors)

        valid_norm_idxs = np.where(norm > 0)[0]

        normed_vectors[valid_norm_idxs] = vectors[valid_norm_idxs] / norm[
            valid_norm_idxs
        ].reshape(-1, 1)

        return normed_vectors

    print("[ERROR][normal::normalize]")
    print("\t vectors dim not valid!")
    print("\t vectors.shape:", vectors.shape)
    return np.array([])
