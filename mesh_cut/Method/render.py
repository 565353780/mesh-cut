import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from typing import Union


def createRandomColors(color_num: int, color_map_id: str = "tab20") -> np.ndarray:
    mode = "fixed-random"

    if mode == "fixed-random":
        rng = np.random.default_rng(seed=42)  # 固定种子
        return rng.random((color_num, 3))

    if mode == "random":
        return np.random.rand(color_num, 3)

    if mode == "cmap":
        color_map = plt.get_cmap(color_map_id)

        return np.array(
            [color_map(i % len(color_map.colors))[:3] for i in range(color_num)]
        )


def toTriangleSoup(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    all_vertices = []
    all_triangles = []

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    for tri in triangles:
        v0, v1, v2 = vertices[tri]

        base_idx = len(all_vertices)
        all_vertices.extend([v0, v1, v2])
        all_triangles.append([base_idx, base_idx + 1, base_idx + 2])

    all_vertices = np.array(all_vertices)
    all_triangles = np.array(all_triangles)

    triangle_soup = o3d.geometry.TriangleMesh()
    triangle_soup.vertices = o3d.utility.Vector3dVector(all_vertices)
    triangle_soup.triangles = o3d.utility.Vector3iVector(all_triangles)
    return triangle_soup


def paintTriangleSoup(
    triangle_soup: o3d.geometry.TriangleMesh,
    triangle_colors: Union[np.ndarray, list],
) -> bool:
    triangles = np.asarray(triangle_soup.triangles)

    vertex_colors = np.zeros_like(np.asarray(triangle_soup.vertices))
    for i, triangle in enumerate(triangles):
        vertex_colors[triangle] = triangle_colors[i]

    triangle_soup.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return True


def renderFaceLabelList(
    vertices: np.ndarray,
    triangles: np.ndarray,
    face_label_list: list,
) -> bool:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    triangle_soup = toTriangleSoup(mesh)

    num_submeshes = len(face_label_list) + 1
    colors = createRandomColors(num_submeshes)
    colors[0][:] = 0.0

    face_colors = np.zeros_like(triangles).astype(np.float64)

    for i, submesh_faces in enumerate(face_label_list):
        face_colors[submesh_faces] = colors[i + 1]

    paintTriangleSoup(triangle_soup, face_colors)

    o3d.visualization.draw_geometries([triangle_soup])
    return True


def renderFaceLabels(
    vertices: np.ndarray, triangles: np.ndarray, face_labels: np.ndarray
) -> bool:
    print("start render face labels...")
    num_submeshes = np.max(face_labels) + 2
    colors = createRandomColors(num_submeshes)
    colors[0][:] = 0.0

    face_colors = colors[face_labels + 1]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    triangle_soup = toTriangleSoup(mesh)

    paintTriangleSoup(triangle_soup, face_colors)

    o3d.visualization.draw_geometries([triangle_soup])
    return True


def renderSubMeshSamplePoints(sub_mesh_sample_points: np.ndarray) -> bool:
    color_num, point_num = sub_mesh_sample_points.shape[:2]

    pcd = o3d.geometry.PointCloud()

    print("start create pcd points...")
    points = sub_mesh_sample_points.reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(points)

    print("start create pcd colors...")
    unique_colors = createRandomColors(color_num)
    colors = np.repeat(unique_colors, point_num, axis=0)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("start render merged pcd...")
    o3d.visualization.draw_geometries([pcd])
    return True
