import numpy as np
from tqdm import trange
from typing import Union

from mesh_cut.Method.triangle import mapToSubMesh, getTriangleAreas
from mesh_cut.Module.smooth_mesh_cutter import SmoothMeshCutter
from mesh_cut.Module.average_mesh_cutter import AverageMeshCutter


class SmoothAverageMeshCutter(SmoothMeshCutter):
    def __init__(
        self,
        mesh_file_path: Union[str, None] = None,
        dist_max: float = 1.0 / 500,
    ):
        super().__init__(mesh_file_path, dist_max)
        return

    def createAverageRegions(
        self,
        region: np.ndarray,
    ) -> list:
        sub_mesh, idx_map = mapToSubMesh(self.vertices, self.triangles, region)

        areas = getTriangleAreas(self.vertices, self.triangles[region])
        area_sum = np.sum(areas)

        bbox = sub_mesh.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        diagonal_length = np.linalg.norm(max_bound - min_bound)
        sphere_radius = diagonal_length / 20.0
        sphere_area = np.pi * sphere_radius**2

        submesh_num = int(area_sum / sphere_area)

        if submesh_num < 2:
            return [region]

        if region.shape[0] / submesh_num < 2.0:
            return [region]

        if submesh_num >= region.shape[0]:
            return region.reshape(-1, 1).tolist()

        average_mesh_cutter = AverageMeshCutter()
        average_mesh_cutter.loadMesh(sub_mesh, float("inf"))

        average_mesh_cutter.cutMesh(submesh_num, 0, 0)

        average_regions = []

        average_face_labels = average_mesh_cutter.face_labels

        for mapped_average_region in average_face_labels:
            if len(mapped_average_region) == 0:
                continue

            average_region = idx_map["t_inv"][mapped_average_region]
            average_regions.append(average_region)

        return average_regions

    def cutMesh(
        self,
        normal_angle_max: float = 10.0,
        points_per_submesh: int = 8192,
    ) -> Union[list, bool]:
        if not super().cutMesh(normal_angle_max, points_per_submesh):
            print("[ERROR][SmoothAverageMeshCutter::cutMesh]")
            print("\t SmoothMeshCutter.cutMesh failed!")
            return False

        new_face_labels = np.ones_like(self.face_labels) * -1

        for i in trange(int(np.max(self.face_labels)) + 1):
            smooth_region = np.where(self.face_labels == i)[0]

            average_regions = self.createAverageRegions(smooth_region)

            for j in range(len(average_regions)):
                new_face_label = int(np.max(new_face_labels)) + 1

                new_face_labels[average_regions[j]] = new_face_label

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
