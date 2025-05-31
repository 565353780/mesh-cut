import sys

sys.path.append("../diff-curvature")

import os
import numpy as np
from mesh_graph_cut.Module.mesh_graph_cutter import MeshGraphCutter


def demo():
    # 设置输入和输出路径
    mesh_file_path = "/home/chli/chLi/Dataset/vae-eval/mesh/000.obj"
    output_dir = "./output"

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置分割参数
    sub_mesh_num = 10  # 分割成10个子网格

    print("Loading mesh from", mesh_file_path)
    mesh_graph_cutter = MeshGraphCutter(mesh_file_path)

    # 可视化原始网格的曲率
    print("Visualizing mesh curvature...")
    mesh_graph_cutter.visualizeCurvature()
    exit()

    # 执行网格切割
    print(f"Cutting mesh into {sub_mesh_num} segments...")
    mesh_graph_cutter.cutMesh(sub_mesh_num)

    # 可视化分割结果
    print("Visualizing mesh segments...")
    segments_output = os.path.join(output_dir, "segments.png")
    mesh_graph_cutter.visualizeSegments(segments_output)

    # 可视化边界顶点
    print("Visualizing boundary vertices...")
    boundaries_output = os.path.join(output_dir, "boundaries.png")
    mesh_graph_cutter.visualizeBoundaries(boundaries_output)

    # 导出分割结果
    print("Exporting mesh segments...")
    segments_dir = os.path.join(output_dir, "segments")
    mesh_graph_cutter.exportSegments(segments_dir)

    # 获取并打印分割信息
    print("\nSegment Information:")
    segments_info = mesh_graph_cutter.getSegmentInfo()
    for info in segments_info:
        print(f"Segment {info['label']}:")
        print(f"  Triangle count: {info['triangle_count']}")
        print(f"  Vertex count: {info['vertex_count']}")
        print(f"  Boundary vertex count: {info['boundary_vertex_count']}")
        print(f"  Interior vertex count: {info['interior_vertex_count']}")
        print(f"  Interior curvature sum: {info['interior_curvature_sum']:.4f}")

    # 计算内部顶点曲率之和的标准差，用于评估分割质量
    curvature_sums = [info["interior_curvature_sum"] for info in segments_info]
    curvature_std = np.std(curvature_sums)
    curvature_mean = np.mean(curvature_sums)
    curvature_cv = (
        curvature_std / abs(curvature_mean) if curvature_mean != 0 else float("inf")
    )

    print("\nEvaluation:")
    print(f"  Mean interior curvature sum: {curvature_mean:.4f}")
    print(f"  Std interior curvature sum: {curvature_std:.4f}")
    print(f"  Coefficient of variation: {curvature_cv:.4f}")
    print(f"  Lower coefficient of variation indicates more balanced segmentation.")

    print("\nDemo completed successfully!")
    return True
