import os
import torch
import numpy as np
import open3d as o3d
from typing import Union

from mesh_graph_cut_cpp import (
    farthest_point_sampling,
    run_parallel_region_growing,
    toSubMeshSamplePoints,
)

from diff_curvature.Module.mesh_curvature import MeshCurvature

from mesh_graph_cut.Method.curvature import toVisiableVertexCurvature
from mesh_graph_cut.Method.render import renderFaceLabels, renderSubMeshSamplePoints


class MeshGraphCutter(object):
    def __init__(self, mesh_file_path: Union[str, None] = None):
        self.mesh_curvature = MeshCurvature()

        self.vertices = None
        self.triangles = None

        self.vertex_curvatures = None
        self.face_curvatures = None

        self.sub_mesh_sample_points = None

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

    def loadMesh(self, mesh_file_path: str) -> bool:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][MeshGraphCutter::loadMesh]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path: ", mesh_file_path)
            return False

        mesh = o3d.io.read_triangle_mesh(mesh_file_path)

        self.vertices = np.asarray(mesh.vertices, dtype=np.float32)
        self.triangles = np.asarray(mesh.triangles, dtype=np.int64)

        if not self.mesh_curvature.loadMesh(self.vertices, self.triangles, "cpu"):
            print("[ERROR][MeshGraphCutter::loadMesh]")
            print("\t loadMesh failed for mesh_curvature!")
            return False

        self.vertex_curvatures = self.mesh_curvature.toMeanV().cpu().numpy()
        self.face_curvatures = self.mesh_curvature.toMeanF().cpu().numpy()
        return True

    def cutMesh(
        self, sub_mesh_num: int = 400, points_per_submesh: int = 8192
    ) -> Union[list, bool]:
        if not self.isValid():
            print("[ERROR][MeshGraphCutter::cutMesh]")
            print("\t mesh is not valid!")
            return False

        fps_idxs = farthest_point_sampling(self.vertices, sub_mesh_num)

        self.face_labels = run_parallel_region_growing(
            self.vertices, self.triangles, fps_idxs, sub_mesh_num
        )

        self.sub_mesh_sample_points = toSubMeshSamplePoints(
            self.vertices, self.triangles, self.face_labels, points_per_submesh
        )
        return True

    def visualizeCurvature(self) -> bool:
        if not self.isValid():
            print("[ERROR][MeshGraphCutter::visualizeCurvature]")
            print("\t mesh is not valid!")
            return False

        curvature_vis = 1.0 - toVisiableVertexCurvature(self.vertex_curvatures)
        curvature_vis = torch.from_numpy(curvature_vis).float()

        self.mesh_curvature.render(curvature_vis)
        return True

    def renderFaceLabels(self) -> bool:
        return renderFaceLabels(self.vertices, self.triangles, self.face_labels)

    def renderSubMeshSamplePoints(self) -> bool:
        return renderSubMeshSamplePoints(self.sub_mesh_sample_points)
