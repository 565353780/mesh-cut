import os
import numpy as np
import open3d as o3d
from typing import Union

from mesh_graph_cut.Method.curvature import toVertexCurvature, toFaceCurvature


class MeshGraphCutter(object):
    def __init__(self, mesh_file_path: Union[str, None] = None):
        self.vertices = None
        self.triangles = None

        self.vertex_curvatures = None
        self.face_curvatures = None

        if mesh_file_path is not None:
            self.loadMesh(mesh_file_path)
        return

    def isValid(self) -> bool:
        if self.vertices is None:
            return False
        if self.triangles is None:
            return False
        if self.vertex_curvatures is None:
            return False
        if self.face_curvatures is None:
            return False

        return True

    def updateCurvatures(self) -> bool:
        self.vertex_curvatures = toVertexCurvature(self.vertices, self.triangles)
        self.face_curvatures = toFaceCurvature(self.vertices, self.triangles)
        return True

    def loadMesh(self, mesh_file_path: str) -> bool:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][MeshGraphCutter::loadMesh]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path: ", mesh_file_path)
            return False

        mesh = o3d.io.read_triangle_mesh(mesh_file_path)

        self.vertices = np.asarray(mesh.vertices, dtype=np.float32)
        self.triangles = np.asarray(mesh.triangles, dtype=np.int32)

        if not self.updateCurvatures():
            print("[ERROR][MeshGraphCutter::loadMesh]")
            print("\t updateCurvatures failed!")
            return False

        return True

    def cutMesh(self, sub_mesh_num: int = 400) -> bool:
        if not self.isValid():
            print("[ERROR][MeshGraphCutter::cutMesh]")
            print("\t mesh is not valid!")
            return False

        if self.triangles.shape[0] <= sub_mesh_num:
            print("[WARN][MeshGraphCutter::cutMesh]")
            print("\t tirangle number <=", sub_mesh_num, "!")
            return True

        print("mesh data:")
        print(self.vertices.shape)
        print(self.triangles.shape)
        print(self.vertex_curvatures.shape)
        print(self.face_curvatures.shape)

        # cut the mesh by curvatures here

        return True
