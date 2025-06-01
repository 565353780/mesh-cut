import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


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
    return True


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
