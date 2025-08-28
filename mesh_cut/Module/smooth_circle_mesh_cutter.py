import numpy as np
from tqdm import trange
from typing import Union

from mesh_cut.Method.triangle import mapToSubMesh
from mesh_cut.Module.smooth_mesh_cutter import SmoothMeshCutter
from mesh_cut.Module.circle_mesh_cutter import CircleMeshCutter


class SmoothCircleMeshCutter(SmoothMeshCutter):
    def __init__(
        self,
        mesh_file_path: Union[str, None] = None,
        dist_max: float = 1.0 / 500,
    ):
        super().__init__(mesh_file_path, dist_max)
        return

    def createCircleRegions(
        self,
        region: np.ndarray,
    ) -> list:
        sub_mesh, idx_map = mapToSubMesh(self.vertices, self.triangles, region)

        circle_mesh_cutter = CircleMeshCutter()
        circle_mesh_cutter.loadMesh(sub_mesh, float("inf"))

        circle_mesh_cutter.cutMesh()

        circle_regions = []

        circle_face_labels = circle_mesh_cutter.face_labels
        for i in range(np.max(circle_face_labels) + 1):
            mapped_circle_region = np.where(circle_face_labels == i)[0]
            circle_region = idx_map["t_inv"][mapped_circle_region]
            circle_regions.append(circle_region)

        return circle_regions

    def cutMesh(
        self,
        normal_angle_max: float = 10.0,
        points_per_submesh: int = 8192,
    ) -> Union[list, bool]:
        if not super().cutMesh(normal_angle_max, points_per_submesh):
            print("[ERROR][SmoothCircleMeshCutter::cutMesh]")
            print("\t SmoothMeshCutter.cutMesh failed!")
            return False

        new_face_labels = np.ones_like(self.face_labels) * -1

        for i in trange(int(np.max(self.face_labels)) + 1):
            smooth_region = np.where(self.face_labels == i)[0]

            circle_regions = self.createCircleRegions(smooth_region)

            for j in range(len(circle_regions)):
                new_face_label = int(np.max(new_face_labels)) + 1

                new_face_labels[circle_regions[j]] = new_face_label

        self.face_labels = new_face_labels

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
