import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from typing import Union
from tqdm import tqdm
from collections import deque
from scipy.spatial import cKDTree
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

from diff_curvature.Module.mesh_curvature import MeshCurvature

from mesh_graph_cut.Method.sample import toFPSIdxs
from mesh_graph_cut.Method.curvature import toVisiableVertexCurvature


def compute_min_radius_cover_all(V, centers):
    """
    V: [N, 3] 所有顶点坐标
    centers: [K] 中心点在V中的索引
    return: 最小半径r，使得所有顶点被至少一个球覆盖
    """
    center_xyz = V[centers]  # [K, 3]

    # 对每个点，找它到所有中心点的距离最小值
    dist = np.linalg.norm(V[:, None, :] - center_xyz[None, :, :], axis=-1)  # [N, K]
    min_dists = np.min(dist, axis=1)  # [N] 每个点到最近中心的距离

    return np.max(min_dists)


def build_vertex_to_face_map(F, n_vertices):
    """构建顶点 -> 所属面片的映射表"""
    v2f = {i: [] for i in range(n_vertices)}
    for fid, face in enumerate(F):
        for v in face:
            v2f[v].append(fid)
    return v2f


def find_connected_faces(center_idx, V, F, tree, v2f, radius):
    """给定一个中心点，查找球内所有与该点连通的面片"""
    center = V[center_idx]
    idx_in_ball = tree.query_ball_point(center, radius)
    idx_in_ball_set = set(idx_in_ball)

    visited_vertices = set()
    connected_faces = set()
    queue = deque([center_idx])

    while queue:
        vid = queue.popleft()
        if vid in visited_vertices or vid not in idx_in_ball_set:
            continue
        visited_vertices.add(vid)

        for fid in v2f[vid]:
            face = F[fid]
            if all(v in idx_in_ball_set for v in face):
                connected_faces.add(fid)
                for v in face:
                    if v not in visited_vertices:
                        queue.append(v)

    return center_idx, list(connected_faces)


def run_parallel_region_growing(V, F, centers, radius, n_jobs=8):
    tree = cKDTree(V)
    v2f = build_vertex_to_face_map(F, len(V))

    with tqdm_joblib(tqdm(desc="Region Growing", total=len(centers))):
        results = Parallel(n_jobs=n_jobs)(
            delayed(find_connected_faces)(idx, V, F, tree, v2f, radius)
            for idx in centers
        )
    return {center: faces for center, faces in results}


def visualize_region_map_by_vertex(V, F, region_map):
    """
    V: [N, 3] 顶点坐标
    F: [M, 3] 面片索引
    region_map: Dict[center_idx -> List[face_idx]]
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    mesh.compute_vertex_normals()

    # 默认每个顶点为灰色
    vertex_colors = np.ones((len(V), 3)) * 0.7

    # 每个区域一个颜色
    region_ids = list(region_map.keys())
    color_map = plt.get_cmap("tab20")  # 最多支持 20 种颜色

    for i, center in enumerate(region_ids):
        face_indices = region_map[center]
        color = color_map(i % 20)[:3]
        # 取这些面的顶点
        face_vertices = F[face_indices].flatten()
        vertex_colors[face_vertices] = color  # 给这些顶点上色

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.visualization.draw_geometries([mesh])


def draw_mesh_with_transparent_spheres(mesh, spheres):
    app = gui.Application.instance
    app.initialize()

    win = app.create_window("Region Growing + Spheres", 1024, 768)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(win.renderer)

    mat_mesh = rendering.MaterialRecord()
    mat_mesh.shader = "defaultLit"

    scene.scene.add_geometry("mesh", mesh, mat_mesh)

    for i, sphere in enumerate(spheres):
        mat_sphere = rendering.MaterialRecord()
        mat_sphere.shader = "defaultLitTransparency"
        mat_sphere.base_color = [1.0, 0.0, 0.0, 0.2]  # 最后一个是 alpha
        mat_sphere.base_roughness = 0.5
        mat_sphere.point_size = 3.0
        scene.scene.add_geometry(f"sphere_{i}", sphere, mat_sphere)

    scene.setup_camera(60, mesh.get_axis_aligned_bounding_box(), [0, 0, 0])
    win.add_child(scene)
    app.run()


def visualize_region_map_with_spheres(V, F, region_map, centers, radius):
    """
    V: 顶点坐标 [N, 3]
    F: 三角面片 [M, 3]
    region_map: Dict[center_idx -> List[face_idx]]
    centers: List[int]  # 中心点的顶点索引
    radius: float  # 球半径
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)

    # 每个顶点默认灰色
    vertex_colors = np.ones((len(V), 3)) * 0.7
    color_map = plt.get_cmap("tab20")
    region_ids = list(region_map.keys())

    for i, center in enumerate(region_ids):
        face_indices = region_map[center]
        color = color_map(i % 20)[:3]
        face_vertices = F[face_indices].flatten()
        vertex_colors[face_vertices] = color

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # 添加透明球体：表示每个中心点影响半径
    spheres = []
    for i, center in enumerate(centers):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=16)
        sphere.translate(V[center])
        sphere.paint_uniform_color(color_map(i % 20)[:3])
        sphere.compute_vertex_normals()
        sphere.compute_triangle_normals()
        sphere.compute_vertex_normals()
        sphere.compute_triangle_normals()
        sphere = sphere.subdivide_midpoint(1)
        sphere = sphere.filter_smooth_simple(1)
        sphere.compute_vertex_normals()
        spheres.append(sphere)

    draw_mesh_with_transparent_spheres(mesh, spheres)
    return True


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

        fps_idxs = toFPSIdxs(self.vertices, sub_mesh_num)

        print("[INFO][MeshGraphCutter::cutMesh]")
        print("\t start compute min radius to cover all vertices...")
        radius = compute_min_radius_cover_all(self.vertices, fps_idxs)

        region_map = run_parallel_region_growing(
            self.vertices, self.triangles, fps_idxs, radius, os.cpu_count()
        )

        visualize_region_map_by_vertex(self.vertices, self.triangles, region_map)

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
