import numpy as np
from typing import Union

from mesh_cut.Method.triangle import getRegionNeighboors, isVertexNeighboorsChainOrLoop
from mesh_cut.Module.base_normal_mesh_cutter import BaseNormalMeshCutter


class CircleMeshCutter(BaseNormalMeshCutter):
    def __init__(
        self,
        mesh_file_path: Union[str, None] = None,
        dist_max: float = 1.0 / 500,
    ):
        super().__init__(mesh_file_path, dist_max)
        return

    def findCircleRegion(
        self,
        seed_face_idx: int,
    ) -> np.ndarray:
        region = set()

        region.add(seed_face_idx)

        while True:
            neighboors = getRegionNeighboors(region, self.face_adjacency_list)

            free_neighboors = []
            for neighbor_face_idx in neighboors:
                if self.face_labels[neighbor_face_idx] != -1:
                    continue

                free_neighboors.append(neighbor_face_idx)

            if len(free_neighboors) == 0:
                break

            # FIXME: need to render to check if it has any bug
            if not isVertexNeighboorsChainOrLoop(self.triangles, free_neighboors):
                break

            for neighbor_face_idx in free_neighboors:
                region.add(neighbor_face_idx)

        return np.array(list(region), dtype=np.int32)

    def cutMesh(self) -> Union[list, bool]:
        if not self.isValid():
            print("[ERROR][CircleMeshCutter::cutMesh]")
            print("\t mesh is not valid!")
            return False

        if not self.updateVertexNeighboors():
            print("[ERROR][CircleMeshCutter::cutMesh]")
            print("\t updateVertexNeighboors failed!")
            return False

        self.face_labels = np.ones_like(self.face_curvatures, dtype=np.int32) * -1
        sorted_face_curvature_idxs = np.argsort(self.face_curvatures)

        for face_idx in sorted_face_curvature_idxs:
            if self.face_labels[face_idx] != -1:
                continue

            circle_region = self.findCircleRegion(face_idx)

            new_face_label = int(np.max(self.face_labels)) + 1

            self.face_labels[circle_region] = new_face_label

            self.renderFaceLabels()
        exit()

        return True
