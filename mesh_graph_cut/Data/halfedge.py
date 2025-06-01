import time

try:
    from mesh_graph_cut_cpp import HalfEdgeStructure as CppHalfEdgeStructure

    CPP_EXTENSION_LOADED = True
except ImportError as e:
    print(e)
    CPP_EXTENSION_LOADED = False


class HalfEdgeStructure:
    def __init__(self, faces, num_vertices):
        self.edge_to_face, self.vertex_to_edges = build_halfedge_structure(
            faces, num_vertices
        )
        self.faces = faces
        self.num_vertices = num_vertices

    def get_vertex_neighbors(self, vertex_index):
        neighbors = set()
        for edge in self.vertex_to_edges[vertex_index]:
            # 获取边的另一个顶点
            if edge[0] == vertex_index:
                neighbors.add(edge[1])
            else:
                neighbors.add(edge[0])
        return list(neighbors)

    def get_vertex_faces(self, vertex_index):
        faces = set()
        for edge in self.vertex_to_edges[vertex_index]:
            for face_idx in self.edge_to_face.get(edge, []):
                faces.add(face_idx)
        return list(faces)

    def get_face_neighbors(self, face_index):
        face = self.faces[face_index]
        neighbors = set()

        # 检查每条边的相邻面片
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = (min(v1, v2), max(v1, v2))

            for neighbor_face in self.edge_to_face.get(edge, []):
                if neighbor_face != face_index:
                    neighbors.add(neighbor_face)

        return list(neighbors)


def build_halfedge_structure(F, n_vertices):
    """构建半边结构以加速邻接查询"""
    # 如果C++扩展可用，使用C++实现
    if CPP_EXTENSION_LOADED:
        start_time = time.time()
        halfedge = CppHalfEdgeStructure(F, n_vertices)
        print(f"[C++] build_halfedge_structure 耗时: {time.time() - start_time:.4f}秒")
        return halfedge

    # 否则使用优化的Python实现
    start_time = time.time()

    # 边到面的映射
    edge_to_face = {}
    # 顶点到边的映射
    vertex_to_edges = [[] for _ in range(n_vertices)]

    for fid, (v1, v2, v3) in enumerate(F):
        # 为每个面片的三条边添加映射
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            edge = (min(a, b), max(a, b))  # 规范化边表示
            if edge in edge_to_face:
                edge_to_face[edge].append(fid)
            else:
                edge_to_face[edge] = [fid]

            # 更新顶点到边的映射
            vertex_to_edges[a].append(edge)
            vertex_to_edges[b].append(edge)

    print(f"[Python] build_halfedge_structure 耗时: {time.time() - start_time:.4f}秒")
    return edge_to_face, vertex_to_edges
