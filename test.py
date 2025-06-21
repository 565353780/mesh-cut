import torch
import numpy as np
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple

from mesh_graph_cut.Method.render import createRandomColors


def toCentersAndRadius(
    mesh_file_path: str, center_num: int, cover_point_num: int
) -> Tuple[np.ndarray, float]:
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)

    region_center_pcd = mesh.sample_points_poisson_disk(center_num)

    region_centers = np.asarray(region_center_pcd.points)

    surface_pcd = mesh.sample_points_poisson_disk(cover_point_num)

    distances = np.asarray(surface_pcd.compute_point_cloud_distance(region_center_pcd))

    radius = np.max(distances)

    if False:
        merged_sphere = o3d.geometry.TriangleMesh()

        color_map = plt.get_cmap("tab20")
        colors = createRandomColors(color_map, center_num)
        for i in range(center_num):
            sphere = sphere = o3d.geometry.TriangleMesh.create_sphere(radius, 20)
            sphere.translate(region_centers[i])
            sphere.paint_uniform_color(colors[i])

            merged_sphere += sphere

        o3d.visualization.draw_geometries([merged_sphere])

    return region_centers, radius


def compute_geodesic_distance(mesh, query_points, r):
    """
    计算从每个查询点出发，测地距离不超过r的区域的点云。
    :param mesh: 输入的三角网格
    :param query_points: 查询点 (N x 3)
    :param r: 半径
    :return: 测地距离范围内的点集 (N x M x 3)
    """
    # 获取网格的顶点和三角形
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # 使用网络图来计算测地距离
    # 将三角形的每条边作为图的边来构建邻接图
    G = nx.Graph()
    num_vertices = len(vertices)
    for tri in triangles:
        for i in range(3):
            for j in range(i + 1, 3):
                p1, p2 = tri[i], tri[j]
                dist = np.linalg.norm(vertices[p1] - vertices[p2])
                G.add_edge(p1, p2, weight=dist)

    # 使用 Dijkstra 计算每个查询点的测地距离
    def dijkstra_from_point(start_idx):
        distances = nx.single_source_dijkstra_path_length(G, start_idx)
        return distances

    # 获取所有查询点的测地距离
    geodesic_distances = []
    for point in query_points:
        start_idx = mesh.closest_point(point)
        distances = dijkstra_from_point(start_idx)
        geodesic_distances.append(distances)

    return geodesic_distances


def uniform_sampling_on_surface(vertices, region_indices, M):
    """
    在网格表面指定区域内均匀采样M个点
    :param vertices: 网格的顶点集合
    :param region_indices: 区域内的顶点索引
    :param M: 采样数量
    :return: 均匀采样的点集合
    """
    region_vertices = vertices[region_indices]
    sampled_points = []

    # 可以使用点的随机采样方法，比如最远点采样（Farthest Point Sampling）
    for _ in range(M):
        sampled_points.append(region_vertices[np.random.choice(len(region_vertices))])

    return np.array(sampled_points)


def geodesic_sampling(mesh, query_points, r, M):
    """
    并行地从每个查询点开始，计算测地距离不超过半径r的区域的M个均匀采样点
    :param mesh: 输入的三角网格
    :param query_points: 查询点 (N x 3)
    :param r: 半径
    :param M: 每个区域的采样数量
    :return: NxMx3的点云集合
    """
    N = len(query_points)
    all_samples = []

    # 获取网格顶点
    vertices = np.asarray(mesh.vertices)

    # 计算每个查询点的测地距离
    geodesic_distances = compute_geodesic_distance(mesh, query_points, r)

    for i in range(N):
        # 获取距离查询点测地距离小于r的顶点索引
        region_indices = [
            idx for idx, dist in geodesic_distances[i].items() if dist <= r
        ]

        # 对区域内的点进行均匀采样
        sampled_points = uniform_sampling_on_surface(vertices, region_indices, M)

        # 存储采样点
        all_samples.append(sampled_points)

    return np.array(all_samples)


# 示例用法
if __name__ == "__main__":
    mesh_file_path = "/Users/chli/chLi/Dataset/Objaverse_82K/trimesh/000-000/000a00944e294f7a94f95d420fdd45eb.obj"
    anchor_num = 4096
    cover_point_num = 20000
    sample_point_num = 1024

    # 加载网格
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)

    query_points, r = toCentersAndRadius(mesh_file_path, anchor_num, cover_point_num)

    query_points_tensor = torch.tensor(query_points, dtype=torch.float32)

    sampled_points = geodesic_sampling(mesh, query_points_tensor, r, sample_point_num)
    print(sampled_points.shape)
