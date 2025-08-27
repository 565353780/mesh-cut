import trimesh
import numpy as np
from tqdm import tqdm
from typing import Union
from collections import deque

from cut_cpp import toSubMeshSamplePoints

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

        self.face_adjacency_list = []

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

        self.vertex_curvatures = self.mesh_curvature.toGaussV()
        self.face_curvatures = self.mesh_curvature.toGaussF()

        self.vertex_curvatures = np.abs(self.vertex_curvatures)
        self.face_curvatures = np.abs(self.face_curvatures)
        return True

    def updateTriangleNeighboors(self) -> bool:
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.triangles)

        face_adjacency = mesh.face_adjacency
        self.face_adjacency_list = [[] for _ in range(self.triangles.shape[0])]

        for i, j in face_adjacency:
            self.face_adjacency_list[i].append(int(j))
            self.face_adjacency_list[j].append(int(i))
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

    def findSmoothRegion(
        self,
        seed_face_idx: int,
        normal_angle_max: float,
    ) -> list:
        visited = set()
        region = set()

        queue = deque()
        queue.append(seed_face_idx)
        visited.add(seed_face_idx)
        region.add(seed_face_idx)

        while queue:
            current_face_idx = queue.popleft()
            for neighbor_face_idx in self.face_adjacency_list[current_face_idx]:
                if neighbor_face_idx in visited:
                    continue

                visited.add(neighbor_face_idx)

                angle = self.toFaceNormalAngle(seed_face_idx, neighbor_face_idx)
                if angle <= normal_angle_max:
                    region.add(neighbor_face_idx)
                    queue.append(neighbor_face_idx)

        return list(region)

    def cutMesh(
        self,
        normal_angle_max: float = 10.0,
        points_per_submesh: int = 8192,
    ) -> Union[list, bool]:
        if not self.isValid():
            print("[ERROR][MeshCutter::cutMesh]")
            print("\t mesh is not valid!")
            return False

        if not self.updateTriangleNeighboors():
            print("[ERROR][NormalMeshCutter::cutMesh]")
            print("\t updateTriangleNeighboors failed!")
            return False

        self.face_labels = np.ones_like(self.face_curvatures, dtype=np.int32) * -1
        sorted_face_curvature_idxs = np.argsort(self.face_curvatures)

        for face_idx in tqdm(sorted_face_curvature_idxs):
            if self.face_labels[face_idx] != -1:
                continue

            new_region = self.findSmoothRegion(face_idx, normal_angle_max)
            new_face_label = int(np.max(self.face_labels)) + 1

            self.face_labels[new_region] = new_face_label

        new_face_labels = []
        for i in range(np.max(self.face_labels) + 1):
            curr_face_idxs = np.where(self.face_labels == i)[0]
            if curr_face_idxs.shape[0] == 0:
                continue

            new_face_labels.append(curr_face_idxs)

        self.face_labels = new_face_labels

        """
        self.sub_mesh_sample_points = toSubMeshSamplePoints(
            torch.from_numpy(self.vertices).to(torch.float32),
            torch.from_numpy(self.triangles).to(torch.int),
            self.face_labels,
            points_per_submesh,
        )
        """
        return True

    def visualizeCurvature(self) -> bool:
        if not self.isValid():
            print("[ERROR][MeshCutter::visualizeCurvature]")
            print("\t mesh is not valid!")
            return False

        curvature_vis = toVisiableVertexCurvature(self.face_curvatures)

        self.mesh_curvature.render(curvature_vis)
        return True
