import os
import torch
import numpy as np
import open3d as o3d
from typing import Union

from cut_cpp import (
    farthest_point_sampling,
    run_parallel_region_growing,
    toSubMeshSamplePoints,
)

from diff_curvature.Module.mesh_curvature import MeshCurvature

from mesh_cut.Method.curvature import toVisiableVertexCurvature
from mesh_cut.Method.render import renderFaceLabels, renderSubMeshSamplePoints


class MeshCutter(object):
    def __init__(self, mesh_file_path: Union[str, None] = None):
        self.mesh_curvature = MeshCurvature()

        self.vertices = None
        self.triangles = None

        self.vertex_normals = None

        self.vertex_curvatures = None
        self.face_curvatures = None

        # cut mesh results
        self.fps_vertex_idxs = None
        self.sub_mesh_sample_points = None

        if mesh_file_path is not None:
            self.loadMesh(mesh_file_path)
        return

    def isValid(self) -> bool:
        if self.vertices is None:
            return False
        if self.triangles is None:
            return False

        if self.vertex_normals is None:
            return False

        if self.vertex_curvatures is None:
            return False
        if self.face_curvatures is None:
            return False

        return True

    def loadMesh(self, mesh_file_path: str) -> bool:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][MeshCutter::loadMesh]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path: ", mesh_file_path)
            return False

        mesh = o3d.io.read_triangle_mesh(mesh_file_path)

        self.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.triangles = np.asarray(mesh.triangles, dtype=np.int32)

        mesh.compute_vertex_normals()
        self.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        return True

    def subdivMesh(self, target_vertex_num: int) -> bool:
        if self.vertices.shape[0] >= target_vertex_num:
            return True

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)

        while len(mesh.vertices) < target_vertex_num:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)

        self.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.triangles = np.asarray(mesh.triangles, dtype=int)

        mesh.compute_vertex_normals()
        self.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

        o3d.io.write_triangle_mesh("../ma-sh/output/subdiv.ply", mesh)
        return True

    def estimateCurvatures(self) -> bool:
        if not self.mesh_curvature.loadMesh(self.vertices, self.triangles, "cpu"):
            print("[ERROR][MeshCutter::estimateCurvatures]")
            print("\t loadMesh failed for mesh_curvature!")
            return False

        self.vertex_curvatures = self.mesh_curvature.toMeanV()
        self.face_curvatures = self.mesh_curvature.toMeanF()
        return True

    def cutMesh(
        self, sub_mesh_num: int = 400, points_per_submesh: int = 8192
    ) -> Union[list, bool]:
        self.subdivMesh(10 * sub_mesh_num)

        self.estimateCurvatures()

        if not self.isValid():
            print("[ERROR][MeshCutter::cutMesh]")
            print("\t mesh is not valid!")
            return False

        self.fps_vertex_idxs = farthest_point_sampling(
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
            self.estimateCurvatures()

        if not self.isValid():
            print("[ERROR][MeshCutter::visualizeCurvature]")
            print("\t mesh is not valid!")
            return False

        curvature_vis = 1.0 - toVisiableVertexCurvature(self.vertex_curvatures)

        self.mesh_curvature.render(curvature_vis)
        return True

    def renderFaceLabels(self) -> bool:
        return renderFaceLabels(self.vertices, self.triangles, self.face_labels)

    def renderSubMeshSamplePoints(self) -> bool:
        return renderSubMeshSamplePoints(self.sub_mesh_sample_points)
