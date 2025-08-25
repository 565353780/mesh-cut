import sys

sys.path.append("../mesh-cut")
sys.path.append("../diff-curvature")

import os
import time
from mesh_cut.Module.mesh_cutter import MeshCutter


def demo():
    # 设置输入和输出路径
    mesh_file_path = "/Users/chli/chLi/Dataset/vae-eval/mesh/000.obj"
    output_dir = "./output"

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置分割参数
    sub_mesh_num = 400
    points_per_submesh = 1024

    print("Loading mesh from", mesh_file_path)
    mesh_cutter = MeshCutter(mesh_file_path)

    # 可视化原始网格的曲率
    # print("Visualizing mesh curvature...")
    # mesh_cutter.visualizeCurvature()

    # 执行网格切割
    print(f"Cutting mesh into {sub_mesh_num} segments...")
    start_time = time.time()
    mesh_cutter.cutMesh(sub_mesh_num, points_per_submesh)
    end_time = time.time()
    print(f"Mesh cutting completed in {end_time - start_time:.2f} seconds")

    print("sub mesh sample points.shape:", mesh_cutter.sub_mesh_sample_points.shape)

    print("Render face labels...")
    mesh_cutter.renderFaceLabels()

    # print("Render sub meshes...")
    # mesh_cutter.renderSubMeshSamplePoints()

    return True
