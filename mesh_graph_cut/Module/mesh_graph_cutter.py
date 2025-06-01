import os
import time
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

from mesh_graph_cut.Data.halfedge import build_halfedge_structure
from mesh_graph_cut.Method.sample import toFPSIdxs
from mesh_graph_cut.Method.curvature import toVisiableVertexCurvature

# 尝试导入C++加速模块
try:
    from mesh_graph_cut_cpp import (
        run_parallel_region_growing as cpp_run_parallel_region_growing,
        find_connected_faces as cpp_find_connected_faces,
        compute_min_radius_cover_all as cpp_compute_min_radius_cover_all,
        build_vertex_to_face_map as cpp_build_vertex_to_face_map,
        HalfEdgeStructure as CppHalfEdgeStructure,
    )

    CPP_EXTENSION_LOADED = True

    print("Using C++ accelerated functions for mesh graph cutting")
except ImportError as e:
    print("C++ extension not available, using Python implementation")
    print(e)
    CPP_EXTENSION_LOADED = False


def compute_min_radius_cover_all(V, centers):
    """
    V: [N, 3] 所有顶点坐标
    centers: [K] 中心点在V中的索引
    return: 最小半径r，使得所有顶点被至少一个球覆盖
    """
    # 如果C++扩展可用，使用C++实现
    if CPP_EXTENSION_LOADED:
        start_time = time.time()
        result = cpp_compute_min_radius_cover_all(V, centers)
        print(
            f"[C++] compute_min_radius_cover_all 耗时: {time.time() - start_time:.4f}秒"
        )
        return result[0]  # C++返回的是一个包含单个值的数组

    # 否则使用优化的Python实现
    start_time = time.time()

    # 优化：使用向量化操作和内存效率更高的实现
    center_xyz = V[centers]  # [K, 3]

    # 使用分块处理以减少内存使用
    chunk_size = 10000  # 根据可用内存调整
    n_vertices = V.shape[0]
    n_chunks = (n_vertices + chunk_size - 1) // chunk_size

    min_dists = np.full(n_vertices, np.inf)

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_vertices)

        # 计算当前块中顶点到所有中心点的距离
        chunk_vertices = V[start_idx:end_idx]
        # 使用广播计算距离矩阵
        chunk_dists = np.sqrt(
            np.sum(
                (chunk_vertices[:, np.newaxis, :] - center_xyz[np.newaxis, :, :]) ** 2,
                axis=2,
            )
        )

        # 更新最小距离
        min_dists[start_idx:end_idx] = np.min(chunk_dists, axis=1)

    result = np.max(min_dists)
    print(
        f"[Python] compute_min_radius_cover_all 耗时: {time.time() - start_time:.4f}秒"
    )
    return result


def build_vertex_to_face_map(F, n_vertices):
    """构建顶点 -> 所属面片的映射表"""
    # 如果C++扩展可用，使用C++实现
    if CPP_EXTENSION_LOADED:
        start_time = time.time()
        result = cpp_build_vertex_to_face_map(F, n_vertices)
        print(f"[C++] build_vertex_to_face_map 耗时: {time.time() - start_time:.4f}秒")
        return result

    # 否则使用优化的Python实现
    start_time = time.time()

    # 优化：使用NumPy操作预分配空间
    v2f = [[] for _ in range(n_vertices)]

    # 使用NumPy的flatten和索引操作加速
    for fid, face in enumerate(F):
        for v in face:
            v2f[v].append(fid)

    print(f"[Python] build_vertex_to_face_map 耗时: {time.time() - start_time:.4f}秒")
    return v2f


def find_connected_faces_optimized(
    center_idx, V, F, tree, v2f, halfedge_structure, radius, face_curvatures=None
):
    """优化版本：给定一个中心点，查找球内所有与该点连通的面片"""
    # 如果C++扩展可用且提供了面片曲率信息，使用C++实现
    if CPP_EXTENSION_LOADED and face_curvatures is not None:
        start_time = time.time()
        # 调用C++实现的find_connected_faces函数
        # 参数：中心点索引，顶点坐标，面片索引，半径，面片曲率
        result = cpp_find_connected_faces(center_idx, V, F, radius, face_curvatures)
        print(
            f"[C++] find_connected_faces for center {center_idx} 耗时: {time.time() - start_time:.4f}秒"
        )
        return center_idx, result

    # 否则使用优化的Python实现
    start_time = time.time()

    center = V[center_idx]
    idx_in_ball = tree.query_ball_point(center, radius)
    idx_in_ball_set = set(idx_in_ball)

    # 使用位图加速成员检查
    in_ball_bitmap = np.zeros(len(V), dtype=bool)
    in_ball_bitmap[idx_in_ball] = True

    visited_vertices = np.zeros(len(V), dtype=bool)
    connected_faces = set()
    queue = deque([center_idx])

    while queue:
        vid = queue.popleft()
        if visited_vertices[vid] or not in_ball_bitmap[vid]:
            continue
        visited_vertices[vid] = True

        # 使用半边结构加速邻接面片查找
        if isinstance(halfedge_structure, CppHalfEdgeStructure):
            # C++实现的半边结构
            vertex_faces = halfedge_structure.get_vertex_faces(vid)
            for fid in vertex_faces:
                face = F[fid]
                # 检查面片的所有顶点是否都在球内
                if (
                    in_ball_bitmap[face[0]]
                    and in_ball_bitmap[face[1]]
                    and in_ball_bitmap[face[2]]
                ):
                    connected_faces.add(fid)
                    # 将未访问的顶点加入队列
                    for v in face:
                        if not visited_vertices[v]:
                            queue.append(v)
        else:
            # Python实现的半边结构
            edge_to_face, vertex_to_edges = halfedge_structure
            for edge in vertex_to_edges[vid]:
                for fid in edge_to_face.get(edge, []):
                    face = F[fid]
                    # 检查面片的所有顶点是否都在球内
                    if (
                        in_ball_bitmap[face[0]]
                        and in_ball_bitmap[face[1]]
                        and in_ball_bitmap[face[2]]
                    ):
                        connected_faces.add(fid)
                        # 将未访问的顶点加入队列
                        for v in face:
                            if not visited_vertices[v]:
                                queue.append(v)

    print(f"[Python] find_connected_faces 耗时: {time.time() - start_time:.4f}秒")
    return center_idx, list(connected_faces)


def run_parallel_region_growing(
    V,
    F,
    centers,
    radius,
    n_jobs=8,
    vertex_curvatures=None,
    face_curvatures=None,
    num_segments=None,
):
    """优化版本：并行区域生长算法"""
    # 如果C++扩展可用且提供了曲率信息，使用C++实现
    if (
        CPP_EXTENSION_LOADED
        and vertex_curvatures is not None
        and face_curvatures is not None
        and num_segments is not None
    ):
        start_time = time.time()
        print("[INFO] 使用C++加速的并行区域生长算法...")
        face_labels = cpp_run_parallel_region_growing(
            V, F, vertex_curvatures, face_curvatures, centers, num_segments
        )
        print(
            f"[C++] run_parallel_region_growing 耗时: {time.time() - start_time:.4f}秒"
        )

        # 将结果转换为与Python实现相同的格式
        result = {}
        for i, center in enumerate(centers[:num_segments]):
            result[center] = []

        # 根据面片标签构建结果
        for face_idx, label in enumerate(face_labels):
            if label < len(centers) and label >= 0:
                center = centers[label]
                if center in result:
                    result[center].append(face_idx)

        return result

    # 否则使用优化的Python实现
    start_time = time.time()
    print("[INFO] 构建KD树和半边结构...")
    tree = cKDTree(V)
    v2f = build_vertex_to_face_map(F, len(V))
    halfedge_structure = build_halfedge_structure(F, len(V))

    print(f"[INFO] 使用{n_jobs}个进程并行处理{len(centers)}个区域...")

    # 使用更高效的并行处理
    with tqdm_joblib(tqdm(desc="Region Growing", total=len(centers))):
        results = Parallel(n_jobs=n_jobs, batch_size=max(1, len(centers) // n_jobs))(
            delayed(find_connected_faces_optimized)(
                idx, V, F, tree, v2f, halfedge_structure, radius, face_curvatures
            )
            for idx in centers
        )

    # 使用字典推导式构建结果
    result = {center: faces for center, faces in results}
    print(
        f"[Python] run_parallel_region_growing 耗时: {time.time() - start_time:.4f}秒"
    )
    return result


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

        # 使用改进的区域生长算法，传递曲率信息以便C++实现使用
        region_map = run_parallel_region_growing(
            self.vertices,
            self.triangles,
            fps_idxs,
            radius,
            os.cpu_count(),
            vertex_curvatures=self.vertex_curvatures,
            face_curvatures=self.face_curvatures,
            num_segments=sub_mesh_num,
        )

        # visualize_region_map_by_vertex(self.vertices, self.triangles, region_map)

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
