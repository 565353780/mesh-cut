import sys

sys.path.append("../mesh-sample")

import os
import time
from mesh_cut.Module.average_mesh_cutter import AverageMeshCutter


def demo():
    # 设置输入和输出路径
    mesh_file_path = "/Users/chli/chLi/Dataset/vae-eval/mesh/000.obj"
    mesh_file_path = "/Users/chli/chLi/Dataset/BitAZ/mesh/BitAZ.ply"
    dist_max = 1.0 / 100
    print_progress = True
    output_dir = "./output"

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置分割参数
    sub_mesh_num = 400
    points_per_submesh = 1024

    print("Loading mesh from", mesh_file_path)
    average_mesh_cutter = AverageMeshCutter(mesh_file_path, dist_max, print_progress)

    # 执行网格切割
    print(f"Cutting mesh into {sub_mesh_num} segments...")
    start_time = time.time()
    average_mesh_cutter.cutMesh(sub_mesh_num, points_per_submesh)
    end_time = time.time()
    print(f"Mesh cutting completed in {end_time - start_time:.2f} seconds")

    print(
        "sub mesh sample points.shape:",
        average_mesh_cutter.sub_mesh_sample_points.shape,
    )

    print("Render face labels...")
    average_mesh_cutter.renderFaceLabels()

    # print("Render sub meshes...")
    # mesh_cutter.renderSubMeshSamplePoints()

    return True
