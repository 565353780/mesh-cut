import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def renderSubMeshSamplePoints(sub_mesh_sample_points: np.ndarray) -> bool:
    color_map = plt.get_cmap("tab20")

    color_num, point_num = sub_mesh_sample_points.shape[:2]

    pcd = o3d.geometry.PointCloud()

    print("start create pcd points...")
    points = sub_mesh_sample_points.reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(points)

    print("start create pcd colors...")
    colors = np.zeros_like(points)
    for i in range(color_num):
        colors[i * point_num : (i + 1) * point_num] = color_map(i % 20)[:3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("start render merged pcd...")
    o3d.visualization.draw_geometries([pcd])
    return True
