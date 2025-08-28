import numpy as np
import open3d as o3d
from typing import Union

from cut_cpp import toSubMeshSamplePoints

from diff_curvature.Module.mesh_curvature import MeshCurvature

from mesh_cut.Method.normal import normalize
from mesh_cut.Method.triangle import createEdgeNeighboors, createVertexNeighboors
from mesh_cut.Method.curvature import toVisiableVertexCurvature
from mesh_cut.Module.base_mesh_cutter import BaseMeshCutter


class BaseNormalMeshCutter(BaseMeshCutter):
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

        self.face_adjacency_list = []

        super().__init__(mesh_file_path, dist_max)
        return

    def isValid(self) -> bool:
        if not super().isValid():
            return False

        if self.vertex_normals.size == 0:
            print("[ERROR][BaseNormalMeshCutter::isValid]")
            print("\t vertex_normals is empty!")
            return False

        if self.triangle_normals.size == 0:
            print("[ERROR][BaseNormalMeshCutter::isValid]")
            print("\t triangle_normals is empty!")
            return False

        if self.vertex_curvatures.size == 0:
            print("[ERROR][BaseNormalMeshCutter::isValid]")
            print("\t vertex_curvatures is empty!")
            return False

        if self.face_curvatures.size == 0:
            print("[ERROR][BaseNormalMeshCutter::isValid]")
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

        self.vertex_curvatures = self.mesh_curvature.toGaussV()
        self.face_curvatures = self.mesh_curvature.toGaussF()

        self.vertex_curvatures = np.abs(self.vertex_curvatures)
        self.face_curvatures = np.abs(self.face_curvatures)
        return True

    def updateEdgeNeighboors(self) -> bool:
        self.face_adjacency_list = createEdgeNeighboors(self.triangles)
        return True

    def updateVertexNeighboors(self) -> bool:
        self.face_adjacency_list = createVertexNeighboors(self.triangles)
        return True

    def loadMesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        dist_max: float = 1.0 / 500,
    ) -> bool:
        if not super().loadMesh(mesh, dist_max):
            return False

        if not self.estimateNormals():
            print("[ERROR][BaseNormalMeshCutter::loadMesh]")
            print("\t estimateNormals failed!")
            return False

        if not self.estimateCurvatures():
            print("[ERROR][BaseNormalMeshCutter::loadMesh]")
            print("\t estimateCurvatures failed!")
            return False

        return True

    def toFaceNormalAngle(
        self,
        face_idx_1: int,
        face_idx_2: int,
    ) -> float:
        if face_idx_1 == face_idx_2:
            return 0.0

        triangle_normal_1 = self.triangle_normals[face_idx_1]
        triangle_normal_2 = self.triangle_normals[face_idx_2]

        cos_value = np.dot(triangle_normal_1, triangle_normal_2)
        cos_value = np.clip(cos_value, -1.0, 1.0)

        angle_rad = np.arccos(cos_value)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def visualizeCurvature(self) -> bool:
        if not self.isValid():
            print("[ERROR][BaseNormalMeshCutter::visualizeCurvature]")
            print("\t mesh is not valid!")
            return False

        curvature_vis = toVisiableVertexCurvature(self.face_curvatures)

        self.mesh_curvature.render(curvature_vis)
        return True
