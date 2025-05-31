import os
import torch
import numpy as np
import open3d as o3d
from typing import Union, List

from diff_curvature.Module.mesh_curvature import MeshCurvature

from mesh_graph_cut.Method.curvature import toVisiableVertexCurvature
from mesh_graph_cut.Method.visualization import (
    visualize_mesh_segments,
    visualize_boundary_vertices,
)


class MeshGraphCutter(object):
    def __init__(self, mesh_file_path: Union[str, None] = None):
        self.mesh_curvature = MeshCurvature()

        self.vertices = None
        self.triangles = None

        self.vertex_curvatures = None
        self.face_curvatures = None

        # 存储切割结果
        self.triangle_labels = None
        self.boundary_vertices = None

        if mesh_file_path is not None:
            self.loadMesh(mesh_file_path)
        return

    def isValid(self) -> bool:
        if self.vertices is None:
            return False
        if self.triangles is None:
            return False
        if self.vertex_curvatures is None:
            return False
        if self.face_curvatures is None:
            return False

        return True

    def loadMesh(self, mesh_file_path: str) -> bool:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][MeshGraphCutter::loadMesh]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path: ", mesh_file_path)
            return False

        mesh = o3d.io.read_triangle_mesh(mesh_file_path)

        self.vertices = np.asarray(mesh.vertices, dtype=np.float32)
        self.triangles = np.asarray(mesh.triangles, dtype=np.int64)

        if not self.mesh_curvature.loadMesh(self.vertices, self.triangles, "cpu"):
            print("[ERROR][MeshGraphCutter::loadMesh]")
            print("\t loadMesh failed for mesh_curvature!")
            return False

        self.vertex_curvatures = self.mesh_curvature.toMeanV().cpu().numpy()
        self.face_curvatures = self.mesh_curvature.toMeanF().cpu().numpy()
        return True

    def cutMesh(self, sub_mesh_num: int = 400) -> bool:
        if not self.isValid():
            print("[ERROR][MeshGraphCutter::cutMesh]")
            print("\t mesh is not valid!")
            return False

        if self.triangles.shape[0] <= sub_mesh_num:
            print("[WARN][MeshGraphCutter::cutMesh]")
            print("\t tirangle number <=", sub_mesh_num, "!")
            return True

        print("mesh data:")
        print("vertices shape:", self.vertices.shape)
        print("triangles shape:", self.triangles.shape)
        print("vertex_curvatures shape:", self.vertex_curvatures.shape)
        print("face_curvatures shape:", self.face_curvatures.shape)

        # 构建三角形邻接图
        print("Building triangle adjacency graph...")
        triangle_graph = self._buildTriangleGraph()

        # 构建图的拉普拉斯矩阵，考虑曲率信息
        print("Building Laplacian matrix with curvature information...")
        laplacian = self._buildLaplacianWithCurvature(triangle_graph)

        # 使用谱聚类进行图割
        print(
            f"Performing spectral clustering to divide mesh into {sub_mesh_num} parts..."
        )
        triangle_labels = self._spectralClustering(laplacian, sub_mesh_num)

        # 优化边界，使高曲率顶点尽可能位于边界上
        print("Optimizing boundaries to place high curvature vertices on boundaries...")
        self._optimizeBoundaries(triangle_labels, triangle_graph)

        # 保存结果
        self.triangle_labels = triangle_labels
        print("Mesh cutting completed successfully.")

        return True

    def _buildTriangleGraph(self):
        """构建三角形邻接图，如果两个三角形共享一条边，则它们相邻"""
        import scipy.sparse as sp
        from collections import defaultdict

        n_triangles = self.triangles.shape[0]

        # 创建边到三角形的映射
        edge_to_triangle = defaultdict(list)

        # 对每个三角形的每条边
        for i in range(n_triangles):
            triangle = self.triangles[i]
            # 三角形的三条边
            edges = [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ]

            # 确保边的顶点索引是有序的
            edges = [tuple(sorted(edge)) for edge in edges]

            # 将三角形添加到每条边的列表中
            for edge in edges:
                edge_to_triangle[edge].append(i)

        # 创建三角形邻接矩阵的数据结构
        rows = []
        cols = []
        data = []

        # 对于每条边，如果它连接两个三角形，则这两个三角形相邻
        for edge, triangles in edge_to_triangle.items():
            if len(triangles) == 2:  # 内部边连接两个三角形
                t1, t2 = triangles
                # 添加双向连接
                rows.extend([t1, t2])
                cols.extend([t2, t1])
                data.extend([1, 1])

        # 创建稀疏邻接矩阵
        triangle_graph = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_triangles, n_triangles)
        )

        return triangle_graph

    def _buildLaplacianWithCurvature(self, triangle_graph):
        """构建考虑曲率信息的拉普拉斯矩阵"""
        import scipy.sparse as sp
        import numpy as np

        n_triangles = self.triangles.shape[0]

        # 计算每个三角形的平均曲率（使用面曲率）
        triangle_curvatures = self.face_curvatures

        # 计算曲率差异权重
        rows = []
        cols = []
        weights = []

        # 获取邻接矩阵的非零元素（即相邻三角形对）
        cx = triangle_graph.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            if i < j:  # 只处理每对三角形一次
                # 计算曲率差异，使用指数函数增强差异
                curv_diff = abs(triangle_curvatures[i] - triangle_curvatures[j])
                # 权重与曲率差异成反比，曲率差异大的边权重小
                weight = np.exp(-5.0 * curv_diff)  # 参数5.0可以调整

                # 添加双向边
                rows.extend([i, j])
                cols.extend([j, i])
                weights.extend([weight, weight])

        # 创建权重邻接矩阵
        W = sp.csr_matrix((weights, (rows, cols)), shape=(n_triangles, n_triangles))

        # 计算度矩阵
        degrees = np.array(W.sum(axis=1)).flatten()
        D = sp.diags(degrees)

        # 计算拉普拉斯矩阵 L = D - W
        L = D - W

        return L

    def _spectralClustering(self, laplacian, n_clusters):
        """使用谱聚类算法进行图割"""
        import numpy as np
        from scipy.sparse.linalg import eigsh
        from sklearn.cluster import KMeans

        n_triangles = laplacian.shape[0]

        # 计算拉普拉斯矩阵的特征值和特征向量
        print("Computing eigenvalues and eigenvectors...")
        # 使用最小的k个特征值/特征向量，k=n_clusters
        eigenvalues, eigenvectors = eigsh(laplacian, k=n_clusters, which="SM")

        # 使用前n_clusters个特征向量进行聚类
        print("Clustering using K-means...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(eigenvectors)

        return labels

    def _optimizeBoundaries(self, triangle_labels, triangle_graph):
        """优化边界，使高曲率顶点尽可能位于边界上"""
        import numpy as np
        from collections import defaultdict

        n_triangles = self.triangles.shape[0]
        n_vertices = self.vertices.shape[0]
        n_clusters = len(np.unique(triangle_labels))

        # 找出所有边界三角形
        boundary_triangles = set()
        cx = triangle_graph.tocoo()

        for i, j in zip(cx.row, cx.col):
            if (
                triangle_labels[i] != triangle_labels[j]
            ):  # 不同标签的三角形之间的边是边界
                boundary_triangles.add(i)
                boundary_triangles.add(j)

        # 创建顶点到三角形的映射
        vertex_to_triangles = defaultdict(list)
        for i in range(n_triangles):
            for v in self.triangles[i]:
                vertex_to_triangles[v].append(i)

        # 找出所有边界顶点
        boundary_vertices = set()
        for v in range(n_vertices):
            # 获取包含该顶点的所有三角形
            v_triangles = vertex_to_triangles[v]
            # 检查这些三角形是否属于不同的聚类
            labels = set(triangle_labels[t] for t in v_triangles)
            if len(labels) > 1:  # 如果顶点连接不同标签的三角形，则它是边界顶点
                boundary_vertices.add(v)

        # 计算每个顶点的曲率绝对值
        abs_curvatures = np.abs(self.vertex_curvatures)

        # 对顶点按曲率绝对值排序（从高到低）
        sorted_vertices = np.argsort(-abs_curvatures)

        # 尝试将高曲率顶点移动到边界上
        max_iterations = 100  # 限制迭代次数
        for iteration in range(max_iterations):
            moved = False

            # 遍历高曲率顶点
            for v in sorted_vertices[
                : int(n_vertices * 0.1)
            ]:  # 只考虑曲率最高的10%顶点
                if v in boundary_vertices:  # 如果已经在边界上，跳过
                    continue

                # 获取包含该顶点的所有三角形
                v_triangles = vertex_to_triangles[v]

                # 检查这些三角形当前的标签
                current_labels = [triangle_labels[t] for t in v_triangles]
                if len(set(current_labels)) > 1:  # 已经是边界顶点
                    boundary_vertices.add(v)
                    continue

                # 获取相邻顶点
                neighbor_vertices = set()
                for t in v_triangles:
                    for vn in self.triangles[t]:
                        if vn != v:
                            neighbor_vertices.add(vn)

                # 检查是否有相邻顶点在边界上
                boundary_neighbors = neighbor_vertices.intersection(boundary_vertices)
                if not boundary_neighbors:  # 没有边界邻居，无法移动到边界
                    continue

                # 选择一个边界邻居，并将当前顶点的三角形移动到该邻居的另一个标签
                for neighbor in boundary_neighbors:
                    # 获取邻居顶点的三角形
                    neighbor_triangles = vertex_to_triangles[neighbor]

                    # 获取邻居三角形的标签
                    neighbor_labels = [triangle_labels[t] for t in neighbor_triangles]
                    unique_labels = set(neighbor_labels)

                    # 如果邻居连接多个标签，选择一个不同于当前顶点标签的标签
                    current_label = current_labels[0]  # 当前顶点的标签
                    for label in unique_labels:
                        if label != current_label:
                            # 将当前顶点的一半三角形移动到新标签
                            triangles_to_move = v_triangles[: len(v_triangles) // 2]
                            for t in triangles_to_move:
                                triangle_labels[t] = label

                            # 将顶点标记为边界顶点
                            boundary_vertices.add(v)
                            moved = True
                            break

                    if moved:
                        break

                if moved:
                    break

            if not moved:  # 如果没有顶点被移动，结束迭代
                break

        # 保存边界顶点集合
        self.boundary_vertices = boundary_vertices

        return triangle_labels

    def visualizeCurvature(self) -> bool:
        if not self.isValid():
            print("[ERROR][MeshGraphCutter::visualizeCurvature]")
            print("\t mesh is not valid!")
            return False

        curvature_vis = 1.0 - toVisiableVertexCurvature(self.vertex_curvatures)
        curvature_vis = torch.from_numpy(curvature_vis).float()

        self.mesh_curvature.render(curvature_vis)
        return True

    def visualizeSegments(self, output_path: Union[str, None] = None) -> bool:
        """可视化网格分割结果"""
        if not self.isValid() or self.triangle_labels is None:
            print("[ERROR][MeshGraphCutter::visualizeSegments]")
            print("\t mesh is not valid or not cut yet!")
            return False

        visualize_mesh_segments(
            self.vertices, self.triangles, self.triangle_labels, output_path
        )
        return True

    def visualizeBoundaries(self, output_path: Union[str, None] = None) -> bool:
        """可视化网格边界顶点"""
        if not self.isValid() or self.boundary_vertices is None:
            print("[ERROR][MeshGraphCutter::visualizeBoundaries]")
            print("\t mesh is not valid or not cut yet!")
            return False

        visualize_boundary_vertices(
            self.vertices, self.triangles, self.boundary_vertices, output_path
        )
        return True

    def exportSegments(self, output_dir: str) -> bool:
        """导出分割结果为多个OBJ文件"""
        if not self.isValid() or self.triangle_labels is None:
            print("[ERROR][MeshGraphCutter::exportSegments]")
            print("\t mesh is not valid or not cut yet!")
            return False

        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 获取唯一的标签
        unique_labels = np.unique(self.triangle_labels)

        # 对每个标签，创建一个子网格
        for label in unique_labels:
            # 获取具有该标签的三角形
            mask = self.triangle_labels == label
            sub_triangles = self.triangles[mask]

            # 创建一个新的网格
            sub_mesh = o3d.geometry.TriangleMesh()
            sub_mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
            sub_mesh.triangles = o3d.utility.Vector3iVector(sub_triangles)

            # 移除未使用的顶点
            sub_mesh.remove_unreferenced_vertices()

            # 保存为OBJ文件
            output_path = os.path.join(output_dir, f"segment_{label}.obj")
            o3d.io.write_triangle_mesh(output_path, sub_mesh)

        print(f"Exported {len(unique_labels)} segments to {output_dir}")
        return True

    def getSegmentInfo(self) -> List[dict]:
        """获取每个分割的信息"""
        if not self.isValid() or self.triangle_labels is None:
            print("[ERROR][MeshGraphCutter::getSegmentInfo]")
            print("\t mesh is not valid or not cut yet!")
            return []

        # 获取唯一的标签
        unique_labels = np.unique(self.triangle_labels)

        # 创建顶点到三角形的映射
        vertex_to_triangles = {}
        for i in range(self.triangles.shape[0]):
            for v in self.triangles[i]:
                if v not in vertex_to_triangles:
                    vertex_to_triangles[v] = []
                vertex_to_triangles[v].append(i)

        # 收集每个分割的信息
        segments_info = []
        for label in unique_labels:
            # 获取具有该标签的三角形
            mask = self.triangle_labels == label
            segment_triangles = self.triangles[mask]

            # 获取该分割中的所有顶点
            segment_vertices = set()
            for triangle in segment_triangles:
                for v in triangle:
                    segment_vertices.add(v)

            # 计算该分割的边界顶点
            boundary_vertices = set()
            for v in segment_vertices:
                # 获取包含该顶点的所有三角形
                v_triangles = vertex_to_triangles[v]
                # 检查这些三角形是否属于不同的分割
                labels = set(self.triangle_labels[t] for t in v_triangles)
                if len(labels) > 1:  # 如果顶点连接不同标签的三角形，则它是边界顶点
                    boundary_vertices.add(v)

            # 计算内部顶点（不包括边界）
            interior_vertices = segment_vertices - boundary_vertices

            # 计算内部顶点的曲率之和
            interior_curvature_sum = np.sum(
                self.vertex_curvatures[list(interior_vertices)]
            )

            # 收集信息
            info = {
                "label": label,
                "triangle_count": len(segment_triangles),
                "vertex_count": len(segment_vertices),
                "boundary_vertex_count": len(boundary_vertices),
                "interior_vertex_count": len(interior_vertices),
                "interior_curvature_sum": float(interior_curvature_sum),
            }
            segments_info.append(info)

        return segments_info
