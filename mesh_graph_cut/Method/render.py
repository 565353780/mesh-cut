import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from numba import jit, prange


@jit(nopython=True, parallel=True)
def flatten_points(arr, out):
    for i in prange(arr.shape[0]):
        for j in range(arr.shape[1]):
            out[i * arr.shape[1] + j] = arr[i, j]
    return out


def renderSubMeshSamplePoints(sub_mesh_sample_points: np.ndarray) -> bool:
    color_map = plt.get_cmap("tab20")

    color_num, point_num = sub_mesh_sample_points.shape[:2]

    pcd = o3d.geometry.PointCloud()

    print("start create pcd points...")
    points = np.empty((color_num * point_num, 3), dtype=np.float64)
    flatten_points(sub_mesh_sample_points, points)
    pcd.points = o3d.utility.Vector3dVector(points)

    print("start create pcd colors...")
    unique_colors = np.array([color_map(i % 20)[:3] for i in range(color_num)])
    colors = np.repeat(unique_colors, point_num, axis=0)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("start render merged pcd...")
    o3d.visualization.draw_geometries([pcd])
    return True
