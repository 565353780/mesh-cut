import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from typing import Tuple

import cut_cpp

from mesh_graph_cut.Method.path import createFileFolder
from mesh_graph_cut.Method.render import createRandomColors


def toValidMesh(mesh_file_path: str, save_mesh_file_path: str) -> bool:
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)

    mesh.remove_non_manifold_edges()

    createFileFolder(save_mesh_file_path)
    o3d.io.write_triangle_mesh(save_mesh_file_path, mesh, write_ascii=True)
    return True


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
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius, 20)
            sphere.translate(region_centers[i])
            sphere.paint_uniform_color(colors[i])

            merged_sphere += sphere

        o3d.visualization.draw_geometries([merged_sphere])

    return region_centers, radius


# 示例用法
if __name__ == "__main__":
    mesh_file_path = "/Users/chli/chLi/Dataset/Objaverse_82K/trimesh/000-000/000a00944e294f7a94f95d420fdd45eb.obj"
    valid_mesh_file_path = "./output/valid_mesh.obj"
    anchor_num = 400
    cover_point_num = 10000
    sample_point_num = 1024

    toValidMesh(mesh_file_path, valid_mesh_file_path)

    mesh = o3d.io.read_triangle_mesh(valid_mesh_file_path)

    region_centers, radius = toCentersAndRadius(
        mesh_file_path, anchor_num, cover_point_num
    )

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius, 20)
    sphere.translate(region_centers[0])

    test_sphere_file_path = "./output/test_sphere.obj"
    createFileFolder(test_sphere_file_path)
    o3d.io.write_triangle_mesh(test_sphere_file_path, sphere, write_ascii=True)

    cut_cpp.cutMesh(valid_mesh_file_path, test_sphere_file_path)

    print("finish!")
