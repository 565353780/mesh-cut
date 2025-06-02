import os
import torch
import numpy as np
import open3d as o3d
from typing import Union

from mesh_graph_cut_cpp import (
    farthest_point_sampling,
    compute_min_radius_cover_all,
    run_parallel_region_growing,
)

from diff_curvature.Module.mesh_curvature import MeshCurvature

from mesh_graph_cut.Method.sample import toFPSIdxs
from mesh_graph_cut.Method.curvature import toVisiableVertexCurvature
from mesh_graph_cut.Method.render import (
    visualize_region_map_by_vertex,
    visualize_region_map_with_spheres,
)


class MeshGraphCutter(object):
    def __init__(self, mesh_file_path: Union[str, None] = None):
        self.mesh_curvature = MeshCurvature()

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

    def cutMesh(self, sub_mesh_num: int = 400) -> bool:
        if not self.isValid():
            print("[ERROR][MeshGraphCutter::cutMesh]")
            print("\t mesh is not valid!")
            return False

        print("mesh data:")
        print("vertices shape:", self.vertices.shape)
        print("triangles shape:", self.triangles.shape)
        print("vertex_curvatures shape:", self.vertex_curvatures.shape)
        print("face_curvatures shape:", self.face_curvatures.shape)

        fps_idxs = farthest_point_sampling(self.vertices, sub_mesh_num)

        print("[INFO][MeshGraphCutter::cutMesh]")
        print("\t start compute min radius to cover all vertices...")
        radius = compute_min_radius_cover_all(self.vertices, fps_idxs)

        # 使用改进的区域生长算法，传递曲率信息以便C++实现使用
        face_labels = run_parallel_region_growing(
            self.vertices, self.triangles, fps_idxs, sub_mesh_num
        )

        # 将结果转换为与Python实现相同的格式
        region_map = {}
        for center in fps_idxs:
            region_map[center] = []

        # 根据面片标签构建结果
        for face_idx, label in enumerate(face_labels):
            if label < len(fps_idxs) and label >= 0:
                center = fps_idxs[label]
                if center in region_map:
                    region_map[center].append(face_idx)

        visualize_region_map_by_vertex(self.vertices, self.triangles, region_map)

        visualize_region_map_with_spheres(
            self.vertices, self.triangles, region_map, fps_idxs, radius
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
