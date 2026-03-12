import numpy as np

from typing import Union, List


class Plane(object):
    def __init__(
        self,
        pos: Union[np.ndarray, List] = [0, 0, 0],
        normal: Union[np.ndarray, List] = [0, 0, 1],
    ) -> None:
        self.pos = np.asarray(pos).astype(np.float64)
        self.normal = np.asarray(normal).astype(np.float64)

        self.update()
        return

    def update(self) -> bool:
        self.normal = self.normal / np.linalg.norm(self.normal)
        return True
