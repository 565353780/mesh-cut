import numpy as np
from tqdm import tqdm
from typing import Union
from collections import deque

from mesh_cut.Module.base_normal_mesh_cutter import BaseNormalMeshCutter


class SmoothMeshCutter(BaseNormalMeshCutter):
    def __init__(
        self,
        mesh_file_path: Union[str, None] = None,
        dist_max: float = float("inf"),
    ):
        super().__init__(mesh_file_path, dist_max)
        return

    def findSmoothRegion(
        self,
        seed_face_idx: int,
        normal_angle_max: float,
    ) -> np.ndarray:
        visited = set()
        region = set()

        queue = deque()
        queue.append(seed_face_idx)
        visited.add(seed_face_idx)
        region.add(seed_face_idx)

        while queue:
            current_face_idx = queue.popleft()
            for neighbor_face_idx in self.face_adjacency_list[current_face_idx]:
                if self.face_labels[neighbor_face_idx] != -1:
                    continue

                if neighbor_face_idx in visited:
                    continue

                visited.add(neighbor_face_idx)

                angle = self.toFaceNormalAngle(seed_face_idx, neighbor_face_idx)
                if angle <= normal_angle_max:
                    region.add(neighbor_face_idx)
                    queue.append(neighbor_face_idx)

        return np.array(list(region), dtype=np.int32)

    def cutMesh(
        self,
        normal_angle_max: float = 10.0,
        points_per_submesh: int = 8192,
    ) -> Union[list, bool]:
        if not self.isValid():
            print("[ERROR][SmoothMeshCutter::cutMesh]")
            print("\t mesh is not valid!")
            return False

        if not self.updateEdgeNeighboors():
            print("[ERROR][SmoothMeshCutter::cutMesh]")
            print("\t updateEdgeNeighboors failed!")
            return False

        self.face_labels = np.ones_like(self.face_curvatures, dtype=np.int32) * -1
        sorted_face_curvature_idxs = np.argsort(self.face_curvatures)

        print("[INFO][SmoothMeshCutter::cutMesh]")
        print("\t start cut mesh by triangle curvature and normal angle...")
        for face_idx in tqdm(sorted_face_curvature_idxs):
            if self.face_labels[face_idx] != -1:
                continue

            smooth_region = self.findSmoothRegion(face_idx, normal_angle_max)

            new_face_label = int(np.max(self.face_labels)) + 1

            self.face_labels[smooth_region] = new_face_label

            # self.renderFaceLabels()

        """
        self.sub_mesh_sample_points = toSubMeshSamplePoints(
            torch.from_numpy(self.vertices).to(torch.float32),
            torch.from_numpy(self.triangles).to(torch.int),
            self.face_labels,
            points_per_submesh,
        )
        """
        return True
