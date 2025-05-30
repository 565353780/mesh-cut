import numpy as np
from scipy.sparse import csr_matrix


def compute_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """计算每个顶点的法向量，使用面积加权平均"""
    # 计算每个三角形的法向量
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # 计算三角形的两条边
    edge1 = v1 - v0
    edge2 = v2 - v0

    # 计算叉积得到法向量（未归一化）
    face_normals_unnorm = np.cross(edge1, edge2)

    # 计算三角形面积（叉积的模长的一半）
    face_areas = 0.5 * np.sqrt(np.sum(face_normals_unnorm**2, axis=1))
    face_areas = np.maximum(face_areas, 1e-10)  # 避免除以零

    # 归一化法向量
    face_normals = face_normals_unnorm / (2.0 * face_areas[:, np.newaxis])

    # 初始化顶点法向量和权重
    vertex_normals = np.zeros_like(vertices)
    vertex_weights = np.zeros(vertices.shape[0])

    # 对每个三角形，将其法向量乘以面积加到其三个顶点上
    for i in range(3):
        # 使用面积作为权重
        weighted_normals = face_normals * face_areas[:, np.newaxis]
        np.add.at(vertex_normals, triangles[:, i], weighted_normals)
        np.add.at(vertex_weights, triangles[:, i], face_areas)

    # 确保权重不为零
    vertex_weights = np.maximum(vertex_weights, 1e-10)

    # 应用权重
    vertex_normals = vertex_normals / vertex_weights[:, np.newaxis]

    # 归一化顶点法向量
    norms = np.sqrt(np.sum(vertex_normals**2, axis=1))[:, np.newaxis]
    norms = np.maximum(norms, 1e-10)  # 避免除以零
    vertex_normals = vertex_normals / norms

    return vertex_normals


def compute_vertex_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """计算每个顶点的Voronoi面积"""
    # 计算每个三角形的面积
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # 计算三角形的两条边
    edge1 = v1 - v0
    edge2 = v2 - v0

    # 计算叉积得到面积的两倍
    face_areas = 0.5 * np.sqrt(np.sum(np.cross(edge1, edge2) ** 2, axis=1))

    # 初始化顶点面积
    vertex_areas = np.zeros(vertices.shape[0])

    # 对每个三角形，将其面积的1/3分配给每个顶点
    for i in range(3):
        np.add.at(vertex_areas, triangles[:, i], face_areas / 3.0)

    return vertex_areas


def compute_cotangent_weights(vertices: np.ndarray, triangles: np.ndarray) -> tuple:
    """计算余切权重和拉普拉斯矩阵，使用更稳定的计算方法"""
    n_vertices = vertices.shape[0]
    n_triangles = triangles.shape[0]

    # 初始化权重矩阵的数据结构
    i_indices = np.zeros(6 * n_triangles, dtype=np.int32)
    j_indices = np.zeros(6 * n_triangles, dtype=np.int32)
    values = np.zeros(6 * n_triangles)

    # 对每个三角形计算余切权重
    for t in range(n_triangles):
        # 获取三角形的顶点索引
        i, j, k = triangles[t]

        # 获取顶点坐标
        vi = vertices[i]
        vj = vertices[j]
        vk = vertices[k]

        # 计算三角形的三条边向量
        eij = vj - vi  # 从i到j的边
        ejk = vk - vj  # 从j到k的边
        eki = vi - vk  # 从k到i的边

        # 计算边的长度平方
        lij_sqr = np.sum(eij**2)
        ljk_sqr = np.sum(ejk**2)
        lki_sqr = np.sum(eki**2)

        # 检查三角形是否退化（边长接近于零）
        if lij_sqr < 1e-10 or ljk_sqr < 1e-10 or lki_sqr < 1e-10:
            # 对于退化的三角形，使用小的默认权重
            cot_i = cot_j = cot_k = 0.0
        else:
            # 使用更稳定的余切计算方法
            # 余切值可以通过点积和叉积的比值计算：cot(θ) = dot(a,b) / |cross(a,b)|

            # 计算点积
            dot_i = -np.sum(eki * ejk)  # 角i处的两条边的点积
            dot_j = -np.sum(eij * eki)  # 角j处的两条边的点积
            dot_k = -np.sum(ejk * eij)  # 角k处的两条边的点积

            # 计算叉积的模长
            cross_i = np.linalg.norm(np.cross(eki, ejk))
            cross_j = np.linalg.norm(np.cross(eij, eki))
            cross_k = np.linalg.norm(np.cross(ejk, eij))

            # 避免除以零，并限制余切值的范围以提高稳定性
            epsilon = 1e-10
            cross_i = max(cross_i, epsilon)
            cross_j = max(cross_j, epsilon)
            cross_k = max(cross_k, epsilon)

            # 计算余切值
            cot_i = dot_i / cross_i
            cot_j = dot_j / cross_j
            cot_k = dot_k / cross_k

            # 限制余切值的范围，避免数值不稳定
            cot_max = 10.0  # 限制最大值
            cot_i = min(max(cot_i, -cot_max), cot_max)
            cot_j = min(max(cot_j, -cot_max), cot_max)
            cot_k = min(max(cot_k, -cot_max), cot_max)

        # 填充权重矩阵的数据
        idx = 6 * t
        i_indices[idx : idx + 6] = [i, j, j, k, k, i]
        j_indices[idx : idx + 6] = [j, i, k, j, i, k]
        values[idx : idx + 6] = [cot_k, cot_k, cot_i, cot_i, cot_j, cot_j]

    # 创建稀疏矩阵
    W = csr_matrix((values, (i_indices, j_indices)), shape=(n_vertices, n_vertices))

    # 计算拉普拉斯矩阵 L = D - W，其中D是度矩阵
    D = csr_matrix(
        (
            np.array(W.sum(axis=1)).flatten(),
            (np.arange(n_vertices), np.arange(n_vertices)),
        ),
        shape=(n_vertices, n_vertices),
    )
    L = D - W

    return W, L


def toVertexCurvature(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """计算每个顶点的平均曲率"""
    # 计算顶点法向量
    vertex_normals = compute_vertex_normals(vertices, triangles)

    # 计算顶点面积
    vertex_areas = compute_vertex_areas(vertices, triangles)

    # 计算余切权重和拉普拉斯矩阵
    _, L = compute_cotangent_weights(vertices, triangles)

    # 计算平均曲率向量 H = L * vertices
    H_vector = L.dot(vertices)

    # 计算平均曲率的大小 |H| = |L * vertices| / (2 * area)
    # 添加一个小的epsilon值避免除以零
    epsilon = 1e-10
    H_magnitude = np.sqrt(np.sum(H_vector**2, axis=1)) / (
        2 * np.maximum(vertex_areas, epsilon)
    )

    # 根据法向量方向确定曲率的符号
    # 如果曲率向量与法向量方向一致，则为凸，否则为凹
    H_sign = np.sign(np.sum(H_vector * vertex_normals, axis=1))

    # 最终的平均曲率
    mean_curvature = H_sign * H_magnitude

    # 应用一些处理以增强曲率差异
    # 1. 去除异常值
    percentile_low, percentile_high = np.percentile(mean_curvature, [2, 98])
    mean_curvature = np.clip(mean_curvature, percentile_low, percentile_high)

    # 2. 应用非线性变换增强对比度
    # 使用双曲正切函数进行非线性变换，保留符号但增强对比度
    scale_factor = 3.0  # 控制变换的强度
    mean_curvature = np.tanh(scale_factor * mean_curvature)

    return mean_curvature


def toFaceCurvature(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """计算每个面的曲率，使用面的三个顶点的平均曲率的平均值"""
    # 计算顶点曲率
    vertex_curvature = toVertexCurvature(vertices, triangles)

    # 对每个面，计算其三个顶点的平均曲率的平均值
    face_curvature = np.zeros(triangles.shape[0])
    for i in range(triangles.shape[0]):
        face_curvature[i] = np.mean(vertex_curvature[triangles[i]])

    return face_curvature
