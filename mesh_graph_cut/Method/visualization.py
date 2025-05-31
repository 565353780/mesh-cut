import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def visualize_mesh_segments(vertices, triangles, triangle_labels, output_path=None):
    """
    可视化网格分割结果

    参数:
    vertices: np.ndarray, 顶点坐标数组
    triangles: np.ndarray, 三角形索引数组
    triangle_labels: np.ndarray, 三角形标签数组
    output_path: str, 可选，输出图像的路径
    """
    # 创建一个Open3D网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # 获取唯一的标签
    unique_labels = np.unique(triangle_labels)
    n_segments = len(unique_labels)

    # 创建一个颜色映射
    cmap = plt.get_cmap("tab20")
    if n_segments > 20:
        # 如果分段数超过20，使用hsv颜色映射
        cmap = plt.get_cmap("hsv")

    # 为每个三角形分配颜色
    triangle_colors = np.zeros((len(triangles), 3))
    for i, label in enumerate(unique_labels):
        # 获取颜色
        color = cmap(i / n_segments)[:3]  # 取RGB部分
        # 将该颜色分配给所有具有该标签的三角形
        triangle_colors[triangle_labels == label] = color

    # 设置网格的颜色
    mesh.triangle_colors = o3d.utility.Vector3dVector(triangle_colors)

    # 计算网格的法向量
    mesh.compute_vertex_normals()

    # 可视化网格
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.background_color = np.array([1, 1, 1])  # 白色背景

    # 设置视图
    vis.update_renderer()
    vis.poll_events()
    vis.update_renderer()

    # 如果指定了输出路径，保存图像
    if output_path is not None:
        vis.capture_screen_image(output_path)

    # 显示可视化结果
    vis.run()
    vis.destroy_window()


def visualize_boundary_vertices(
    vertices, triangles, boundary_vertices, output_path=None
):
    """
    可视化边界顶点

    参数:
    vertices: np.ndarray, 顶点坐标数组
    triangles: np.ndarray, 三角形索引数组
    boundary_vertices: set, 边界顶点索引集合
    output_path: str, 可选，输出图像的路径
    """
    # 创建一个Open3D网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # 设置顶点颜色，边界顶点为红色，其他顶点为灰色
    vertex_colors = np.ones((len(vertices), 3)) * 0.8  # 灰色
    for v in boundary_vertices:
        vertex_colors[v] = [1, 0, 0]  # 红色

    # 设置网格的颜色
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # 计算网格的法向量
    mesh.compute_vertex_normals()

    # 可视化网格
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.background_color = np.array([1, 1, 1])  # 白色背景
    opt.point_size = 5.0  # 增大点的大小

    # 设置视图
    vis.update_renderer()
    vis.poll_events()
    vis.update_renderer()

    # 如果指定了输出路径，保存图像
    if output_path is not None:
        vis.capture_screen_image(output_path)

    # 显示可视化结果
    vis.run()
    vis.destroy_window()
