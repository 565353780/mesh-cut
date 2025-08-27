import os
import numpy as np
import open3d as o3d
from typing import Union

from mesh_sample.Module.mesh_subdiver import MeshSubdiver

from mesh_cut.Method.render import renderFaceLabels, renderSubMeshSamplePoints


class BaseMeshCutter(object):
    def __init__(
        self,
        mesh_file_path: Union[str, None] = None,
        dist_max: float = 1.0 / 500,
    ):
        self.vertices = np.array([])
        self.triangles = np.array([])

        # cut mesh results
        self.center_vertex_idxs = np.array([])
        self.face_labels = np.array([])
        self.sub_mesh_sample_points = np.array([])

        if mesh_file_path is not None:
            self.loadMesh(mesh_file_path, dist_max)
        return

    def isValid(self) -> bool:
        if self.vertices.size == 0:
            print("[ERROR][BaseMeshCutter::isValid]")
            print("\t vertices is empty!")
            return False

        if self.triangles.size == 0:
            print("[ERROR][BaseMeshCutter::isValid]")
            print("\t triangles is empty!")
            return False

        return True

    def loadMesh(
        self,
        mesh_file_path: str,
        dist_max: float = 1.0 / 500,
    ) -> bool:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][BaseMeshCutter::loadMesh]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path: ", mesh_file_path)
            return False

        mesh = o3d.io.read_triangle_mesh(mesh_file_path)

        mesh_subdiver = MeshSubdiver(mesh, dist_max)
        subdiv_mesh = mesh_subdiver.createSubdivMesh()

        self.vertices = np.asarray(subdiv_mesh.vertices, dtype=np.float64)
        self.triangles = np.asarray(subdiv_mesh.triangles, dtype=np.int32)
        return True

    def subdivMesh(self, target_vertex_num: int) -> bool:
        if self.vertices.shape[0] >= target_vertex_num:
            return True

        mesh = self.toO3DMesh()

        while len(mesh.vertices) < target_vertex_num:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)

        self.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.triangles = np.asarray(mesh.triangles, dtype=int)
        return True

    def toO3DMesh(self) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)

        return mesh

    def renderFaceLabels(self) -> bool:
        return renderFaceLabels(self.vertices, self.triangles, self.face_labels)

    def renderSubMeshSamplePoints(self) -> bool:
        return renderSubMeshSamplePoints(self.sub_mesh_sample_points)
