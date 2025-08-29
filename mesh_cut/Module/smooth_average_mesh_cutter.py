import numpy as np
import open3d as o3d
from tqdm import trange
from typing import Union, Tuple

from mesh_cut.Method.triangle import mapToSubMesh, getTriangleAreas
from mesh_cut.Module.smooth_mesh_cutter import SmoothMeshCutter
from mesh_cut.Module.average_mesh_cutter import AverageMeshCutter


class SmoothAverageMeshCutter(SmoothMeshCutter):
    def __init__(
        self,
        mesh_file_path: Union[str, None] = None,
        smooth_dist_max: float = float("inf"),
        average_dist_max: float = 1.0 / 100,
    ):
        self.average_dist_max = average_dist_max

        super().__init__(mesh_file_path, smooth_dist_max)
        return

    def createAverageRegions(
        self,
        region: np.ndarray,
    ) -> Tuple[o3d.geometry.TriangleMesh, list]:
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
            return sub_mesh, [list(range(region.shape[0]))]

        average_mesh_cutter = AverageMeshCutter(print_progress=False)
        average_mesh_cutter.loadMesh(sub_mesh, self.average_dist_max)

        average_mesh_cutter.cutMesh(submesh_num, 0, 10)

        subdiv_mesh = average_mesh_cutter.toO3DMesh()
        average_face_labels = average_mesh_cutter.face_labels

        return subdiv_mesh, average_face_labels

    def cutMesh(
        self,
        normal_angle_max: float = 10.0,
        points_per_submesh: int = 8192,
    ) -> Union[list, bool]:
        if not super().cutMesh(normal_angle_max, points_per_submesh):
            print("[ERROR][SmoothAverageMeshCutter::cutMesh]")
            print("\t SmoothMeshCutter.cutMesh failed!")
            return False

        merge_mesh = o3d.geometry.TriangleMesh()
        mrege_face_labels_list = []

        for i in trange(int(np.max(self.face_labels)) + 1):
            smooth_region = np.where(self.face_labels == i)[0]

            subdiv_mesh, average_regions = self.createAverageRegions(smooth_region)

            start_triangle_idx = np.asarray(merge_mesh.triangles).shape[0]

            merge_mesh += subdiv_mesh

            for average_region in average_regions:
                mapped_region = [
                    region_idx + start_triangle_idx for region_idx in average_region
                ]
                mrege_face_labels_list.append(mapped_region)

        self.vertices = np.asarray(merge_mesh.vertices, dtype=np.float64)
        self.triangles = np.asarray(merge_mesh.triangles, dtype=np.int32)
        self.face_labels = mrege_face_labels_list

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
