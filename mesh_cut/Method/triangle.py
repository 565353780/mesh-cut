import numpy as np
import open3d as o3d
from typing import Tuple
from collections import defaultdict


def createEdgeNeighboors(triangles: np.ndarray) -> list:
    edge_to_faces = defaultdict(list)

    for face_index, tri in enumerate(triangles):
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_index)

    adjacency_list = [[] for _ in range(len(triangles))]

    for faces in edge_to_faces.values():
        if len(faces) == 2:
            a, b = faces
            adjacency_list[a].append(b)
            adjacency_list[b].append(a)

    return adjacency_list


def createVertexNeighboors(triangles: np.ndarray) -> list:
    vertex_to_faces = defaultdict(list)

    for face_index, tri in enumerate(triangles):
        for vertex_idx in tri:
            vertex_to_faces[vertex_idx].append(face_index)

    neighbors = []
    for face_idx, tri in enumerate(triangles):
        neighbor_faces = set()
        for v in tri:
            neighbor_faces.update(vertex_to_faces[v])
        neighbor_faces.discard(face_idx)
        neighbors.append(list(neighbor_faces))

    return neighbors


def mapToSubMesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_idxs: np.ndarray,
) -> Tuple[o3d.geometry.TriangleMesh, dict]:
    idx_map = {
        "v": {},
        "v_inv": [],
        "t": {},
        "t_inv": [],
    }

    idx_map["t_inv"] = triangle_idxs

    sub_mesh_triangles = triangles[triangle_idxs]

    unique_vertex_idxs = np.unique(sub_mesh_triangles.flatten())
    idx_map["v_inv"] = unique_vertex_idxs

    for i, vertex_idx in enumerate(unique_vertex_idxs):
        idx_map["v"][vertex_idx] = i

    mapped_vertices = vertices[unique_vertex_idxs]
    mapped_triangles = []
    for i, triangle_idx in enumerate(triangle_idxs):
        idx_map["t"][triangle_idx] = i

        mapped_triangle = [
            idx_map["v"][vertex_idx] for vertex_idx in triangles[triangle_idx]
        ]

        mapped_triangles.append(mapped_triangle)

    sub_mesh = o3d.geometry.TriangleMesh()
    sub_mesh.vertices = o3d.utility.Vector3dVector(mapped_vertices)
    sub_mesh.triangles = o3d.utility.Vector3iVector(mapped_triangles)

    return sub_mesh, idx_map
