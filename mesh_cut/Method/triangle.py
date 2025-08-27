import numpy as np
from collections import defaultdict


def createTriangleNeighboors(triangles: np.ndarray) -> list:
    edge_to_faces = defaultdict(list)

    # 遍历所有面，提取每条边
    for face_index, tri in enumerate(triangles):
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_index)

    adjacency_list = [[] for _ in range(len(triangles))]

    # 第三步：遍历边，填充相邻面索引
    for faces in edge_to_faces.values():
        if len(faces) == 2:
            a, b = faces
            adjacency_list[a].append(b)
            adjacency_list[b].append(a)

    return adjacency_list
