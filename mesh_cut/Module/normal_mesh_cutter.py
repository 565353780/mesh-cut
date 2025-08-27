import torch
import numpy as np
from typing import Union

from cut_cpp import (
    farthest_point_sampling,
    run_parallel_region_growing,
    toSubMeshSamplePoints,
)

from diff_curvature.Module.mesh_curvature import MeshCurvature

from mesh_cut.Method.normal import normalize
from mesh_cut.Method.curvature import toVisiableVertexCurvature
from mesh_cut.Module.base_mesh_cutter import BaseMeshCutter


class NormalMeshCutter(BaseMeshCutter):
    def __init__(
        self,
        mesh_file_path: Union[str, None] = None,
        dist_max: float = 1.0 / 500,
    ):
        self.mesh_curvature = MeshCurvature()

        self.vertex_curvatures = np.array([])
        self.face_curvatures = np.array([])

        self.vertex_normals = np.array([])
        self.triangle_normals = np.array([])

        super().__init__(mesh_file_path, dist_max)
        return

    def isValid(self) -> bool:
        if not super().isValid():
            return False

        if self.vertex_normals.size == 0:
            print("[ERROR][NormalMeshCutter::isValid]")
            print("\t vertex_normals is empty!")
            return False

        if self.triangle_normals.size == 0:
            print("[ERROR][NormalMeshCutter::isValid]")
            print("\t triangle_normals is empty!")
            return False

        if self.vertex_curvatures.size == 0:
            print("[ERROR][NormalMeshCutter::isValid]")
            print("\t vertex_curvatures is empty!")
            return False

        if self.face_curvatures.size == 0:
            print("[ERROR][NormalMeshCutter::isValid]")
            print("\t face_curvatures is empty!")
            return False

        return True

    def estimateNormals(self) -> bool:
        mesh = self.toO3DMesh()

        mesh.compute_vertex_normals()
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        self.vertex_normals = normalize(vertex_normals)

        mesh.compute_triangle_normals()
        triangle_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
        self.triangle_normals = normalize(triangle_normals)
        return True

    def estimateCurvatures(self) -> bool:
        if not self.mesh_curvature.loadMesh(self.vertices, self.triangles, "cpu"):
            print("[ERROR][MeshCutter::estimateCurvatures]")
            print("\t loadMesh failed for mesh_curvature!")
            return False

        self.vertex_curvatures = self.mesh_curvature.toMeanV()
        self.face_curvatures = self.mesh_curvature.toMeanF()
        return True

    def loadMesh(
        self,
        mesh_file_path: str,
        dist_max: float = 1.0 / 500,
    ) -> bool:
        if not super().loadMesh(mesh_file_path, dist_max):
            return False

        if not self.estimateNormals():
            print("[ERROR][NormalMeshCutter::loadMesh]")
            print("\t estimateNormals failed!")
            return False

        if not self.estimateCurvatures():
            print("[ERROR][NormalMeshCutter::loadMesh]")
            print("\t estimateCurvatures failed!")
            return False

        return True

    def cutMesh(
        self,
        normal_angle_max: float = 10.0,
        points_per_submesh: int = 8192,
    ) -> Union[list, bool]:
        if not self.isValid():
            print("[ERROR][MeshCutter::cutMesh]")
            print("\t mesh is not valid!")
            return False

        self.face_labels = np.ones_like(self.face_curvatures, dtype=np.int32) * -1
        sorted_face_curvature_idxs = np.argsort(self.face_curvatures)

        self.center_vertex_idxs = farthest_point_sampling(
            torch.from_numpy(self.vertices).to(torch.float32), sub_mesh_num
        )

        self.face_labels = run_parallel_region_growing(
            self.vertices,
            self.triangles,
            self.fps_vertex_idxs.numpy(),
            sub_mesh_num,
        )

        self.sub_mesh_sample_points = toSubMeshSamplePoints(
            torch.from_numpy(self.vertices).to(torch.float32),
            torch.from_numpy(self.triangles).to(torch.int),
            self.face_labels,
            points_per_submesh,
        )
        return True

    def visualizeCurvature(self) -> bool:
        if not self.isValid():
            print("[ERROR][MeshCutter::visualizeCurvature]")
            print("\t mesh is not valid!")
            return False

        curvature_vis = toVisiableVertexCurvature(self.vertex_curvatures)

        self.mesh_curvature.render(curvature_vis)
        return True
