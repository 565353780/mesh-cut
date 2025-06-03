import sys

sys.path.append("../diff-curvature")

import os
import time
from mesh_graph_cut.Module.mesh_graph_cutter import MeshGraphCutter


def demo():
    # 设置输入和输出路径
    mesh_file_path = "/home/chli/chLi/Dataset/vae-eval/mesh/000.obj"
    output_dir = "./output"

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置分割参数
    sub_mesh_num = 4000
    points_per_submesh = 8192

    print("Loading mesh from", mesh_file_path)
    mesh_graph_cutter = MeshGraphCutter(mesh_file_path)

    # 可视化原始网格的曲率
    # print("Visualizing mesh curvature...")
    # mesh_graph_cutter.visualizeCurvature()

    # 执行网格切割
    print(f"Cutting mesh into {sub_mesh_num} segments...")
    start_time = time.time()
    mesh_graph_cutter.cutMesh(sub_mesh_num, points_per_submesh)
    end_time = time.time()
    print(f"Mesh cutting completed in {end_time - start_time:.2f} seconds")

    print(
        "sub mesh sample points.shape:", mesh_graph_cutter.sub_mesh_sample_points.shape
    )

    print("Render face labels...")
    mesh_graph_cutter.renderFaceLabels()

    print("Render sub meshes...")
    mesh_graph_cutter.renderSubMeshSamplePoints()

    return True
