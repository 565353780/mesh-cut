import sys

sys.path.append("../mesh-sample")
sys.path.append("../diff-curvature")

import os
import time
from mesh_cut.Module.smooth_mesh_cutter import SmoothMeshCutter


def demo():
    # 设置输入和输出路径
    mesh_file_path = "/Users/chli/chLi/Dataset/vae-eval/mesh/000.obj"
    mesh_file_path = "/Users/chli/chLi/Dataset/Famous/bunny-v2.ply"
    # mesh_file_path = "/Users/chli/chLi/Dataset/BitAZ/mesh/BitAZ.ply"
    dist_max = 1.0 / 200
    dist_max = float("inf")
    normal_angle_max = 30.0
    output_dir = "./output"

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置分割参数
    points_per_submesh = 1024

    print("Loading mesh from", mesh_file_path)
    smooth_mesh_cutter = SmoothMeshCutter(mesh_file_path, dist_max)

    # 可视化原始网格的曲率
    # print("Visualizing mesh curvature...")
    # normal_mesh_cutter.visualizeCurvature()
    # exit()

    # 执行网格切割
    print("Cutting mesh into adaptive segments...")
    start_time = time.time()
    smooth_mesh_cutter.cutMesh(normal_angle_max, points_per_submesh)
    end_time = time.time()
    print(f"Mesh cutting completed in {end_time - start_time:.2f} seconds")

    print(
        "sub mesh sample points.shape:", smooth_mesh_cutter.sub_mesh_sample_points.shape
    )

    print("Render face labels...")
    smooth_mesh_cutter.renderFaceLabels()

    # print("Render sub meshes...")
    # normal_mesh_cutter.renderSubMeshSamplePoints()

    return True
