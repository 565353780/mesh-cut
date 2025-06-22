import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def createRandomColors(color_map, color_num: int) -> np.ndarray:
    return np.array(
        [color_map(i % len(color_map.colors))[:3] for i in range(color_num)]
    )


def renderFaceLabels(vertices, triangles, face_labels) -> bool:
    color_map = plt.get_cmap("tab20")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    num_submeshes = len(face_labels)
    colors = createRandomColors(color_map, num_submeshes)

    face_colors = np.zeros_like(triangles).astype(np.float64)

    for i, submesh_faces in enumerate(face_labels):
        face_colors[submesh_faces] = colors[i]

    vertex_colors = np.zeros_like(vertices)
    vertex_count = np.zeros(vertices.shape[0])

    for i, triangle in enumerate(triangles):
        for vertex_idx in triangle:
            vertex_colors[vertex_idx] += face_colors[i]
            vertex_count[vertex_idx] += 1

    vertex_count = np.maximum(vertex_count, 1)[:, np.newaxis]
    vertex_colors = vertex_colors / vertex_count

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    o3d.visualization.draw_geometries([mesh])
    return True


def renderSubMeshSamplePoints(sub_mesh_sample_points: np.ndarray) -> bool:
    color_map = plt.get_cmap("tab20")

    color_num, point_num = sub_mesh_sample_points.shape[:2]

    pcd = o3d.geometry.PointCloud()

    print("start create pcd points...")
    points = sub_mesh_sample_points.reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(points)

    print("start create pcd colors...")
    unique_colors = createRandomColors(color_map, color_num)
    colors = np.repeat(unique_colors, point_num, axis=0)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("start render merged pcd...")
    o3d.visualization.draw_geometries([pcd])
    return True
