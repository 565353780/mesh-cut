import trimesh
import numpy as np
import open3d as o3d

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union
from mesh_cut.Data.plane import Plane


def _merge_close_points(
    points: np.ndarray,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """合并距离小于 tol 的重复点。

    返回: (unique_points, inverse_indices)
        unique_points: (n, 3) 去重后的点
        inverse_indices: (m,) 原始点到去重点的索引映射
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    groups = tree.query_ball_tree(tree, r=tol)

    label = np.full(len(points), -1, dtype=np.int64)
    unique_pts = []
    current_label = 0
    for i, group in enumerate(groups):
        if label[i] >= 0:
            continue
        for j in group:
            label[j] = current_label
        unique_pts.append(points[i].copy())
        current_label += 1

    return np.array(unique_pts), label


@dataclass
class BoundaryLoopMatch:
    positive_loop_idx: int
    negative_loop_idx: int
    transport_cost: float


@dataclass
class CutResult:
    meshes: List[trimesh.Trimesh] = field(default_factory=list)
    boundary_loops: List[List[np.ndarray]] = field(default_factory=list)
    matched_loops: List[BoundaryLoopMatch] = field(default_factory=list)


class PlaneMeshCutter(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def _extract_boundary_loops_from_segments(
        segments: np.ndarray,
        tol: float = 1e-8,
    ) -> List[np.ndarray]:
        """从 line segments 中提取闭合边界环。

        segments: (m, 2, 3) float, 每条线段有两个3D端点。
        返回: 每个环是一个 (k, 3) 的顶点坐标数组。
        """
        if len(segments) == 0:
            return []

        all_pts = segments.reshape(-1, 3)
        unique_pts, inverse = _merge_close_points(all_pts, tol)

        edge_indices = inverse.reshape(-1, 2)

        adj: Dict[int, List[int]] = defaultdict(list)
        for e in edge_indices:
            a, b = int(e[0]), int(e[1])
            if a == b:
                continue
            adj[a].append(b)
            adj[b].append(a)

        visited_edges = set()
        loops: List[np.ndarray] = []

        for start_v in adj:
            for neighbor in adj[start_v]:
                edge_key = (min(start_v, neighbor), max(start_v, neighbor))
                if edge_key in visited_edges:
                    continue

                loop = [start_v]
                prev, curr = start_v, neighbor
                visited_edges.add(edge_key)

                while curr != start_v:
                    loop.append(curr)
                    next_v = None
                    for n in adj[curr]:
                        ek = (min(curr, n), max(curr, n))
                        if ek not in visited_edges:
                            next_v = n
                            visited_edges.add(ek)
                            break
                    if next_v is None:
                        break
                    prev, curr = curr, next_v

                if curr == start_v and len(loop) >= 3:
                    loops.append(unique_pts[np.array(loop)])

        return loops

    @staticmethod
    def _loop_centroid_from_coords(
        loop: np.ndarray,
    ) -> np.ndarray:
        return loop.mean(axis=0)

    @staticmethod
    def _match_boundary_loops(
        positive_loops: List[np.ndarray],
        negative_loops: List[np.ndarray],
    ) -> List[BoundaryLoopMatch]:
        if len(positive_loops) == 0 or len(negative_loops) == 0:
            return []

        pos_centroids = np.array([
            PlaneMeshCutter._loop_centroid_from_coords(lp)
            for lp in positive_loops
        ])
        neg_centroids = np.array([
            PlaneMeshCutter._loop_centroid_from_coords(lp)
            for lp in negative_loops
        ])

        cost_matrix = np.linalg.norm(
            pos_centroids[:, None, :] - neg_centroids[None, :, :], axis=2
        )

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        for r, c in zip(row_ind, col_ind):
            matches.append(BoundaryLoopMatch(
                positive_loop_idx=int(r),
                negative_loop_idx=int(c),
                transport_cost=float(cost_matrix[r, c]),
            ))

        return matches

    @staticmethod
    def _split_face(
        vertices: np.ndarray,
        face: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
        dots: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """将一个被平面切割的三角面拆分为正/负侧的子三角形。

        三角面的三个顶点被分为 2+1 两组。
        同侧两个顶点与线段两端构成四边形 -> 拆成两个三角形。
        另一侧单个顶点与线段两端构成一个三角形。

        返回: (positive_tris, negative_tris, seg_start, seg_end)
            其中 tris 中每个元素是 (3, 3) 的顶点坐标数组。
        """
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        d0, d1, d2 = dots[face[0]], dots[face[1]], dots[face[2]]

        signs = np.array([d0, d1, d2])
        tri_verts = np.array([v0, v1, v2])

        pos_mask = signs > 0
        neg_mask = signs < 0
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        if n_pos == 0 or n_neg == 0:
            # 不该发生，但安全处理：全部归到主侧
            if signs.sum() >= 0:
                return [tri_verts], [], seg_start, seg_end
            else:
                return [], [tri_verts], seg_start, seg_end

        s0, s1 = seg_start, seg_end

        if n_pos == 2 and n_neg == 1:
            alone_idx = np.where(neg_mask)[0][0]
            pair_idxs = np.where(pos_mask)[0]
        elif n_neg == 2 and n_pos == 1:
            alone_idx = np.where(pos_mask)[0][0]
            pair_idxs = np.where(neg_mask)[0]
        else:
            # 某个顶点恰好在平面上(dot==0)的情况
            zero_mask = signs == 0
            if zero_mask.sum() == 1:
                on_plane_idx = np.where(zero_mask)[0][0]
                other_idxs = np.where(~zero_mask)[0]
                if signs[other_idxs[0]] > 0 and signs[other_idxs[1]] > 0:
                    return [tri_verts], [], seg_start, seg_end
                elif signs[other_idxs[0]] < 0 and signs[other_idxs[1]] < 0:
                    return [], [tri_verts], seg_start, seg_end
                else:
                    # 1个在平面上, 另外2个分属两侧 => 交线退化为一个点
                    pos_idx = other_idxs[0] if signs[other_idxs[0]] > 0 else other_idxs[1]
                    neg_idx = other_idxs[0] if signs[other_idxs[0]] < 0 else other_idxs[1]
                    return (
                        [np.array([tri_verts[on_plane_idx], tri_verts[pos_idx], s0])],
                        [np.array([tri_verts[on_plane_idx], tri_verts[neg_idx], s0])],
                        seg_start, seg_end,
                    )
            # 多个顶点在平面上
            if signs.sum() >= 0:
                return [tri_verts], [], seg_start, seg_end
            else:
                return [], [tri_verts], seg_start, seg_end

        alone_v = tri_verts[alone_idx]
        pair_v0 = tri_verts[pair_idxs[0]]
        pair_v1 = tri_verts[pair_idxs[1]]

        edge_len_0 = np.linalg.norm(pair_v0 - alone_v)
        edge_len_1 = np.linalg.norm(pair_v1 - alone_v)

        # s0 在 alone-pair_v0 边上 iff |s0-alone| + |s0-pair_v0| ≈ edge_len_0
        err_s0_on_e0 = abs(
            np.linalg.norm(s0 - alone_v) + np.linalg.norm(s0 - pair_v0) - edge_len_0
        )
        err_s0_on_e1 = abs(
            np.linalg.norm(s0 - alone_v) + np.linalg.norm(s0 - pair_v1) - edge_len_1
        )

        if err_s0_on_e0 <= err_s0_on_e1:
            near_p0, near_p1 = s0, s1
        else:
            near_p0, near_p1 = s1, s0

        orig_normal = np.cross(v1 - v0, v2 - v0)

        # 四边形: pair_v0, near_p0, near_p1, pair_v1
        quad_tri_0 = np.array([pair_v0, near_p0, pair_v1])
        quad_tri_1 = np.array([near_p0, near_p1, pair_v1])

        # 单顶点三角形: alone_v, near_p0, near_p1
        alone_tri = np.array([alone_v, near_p0, near_p1])

        def _align_winding(tri: np.ndarray) -> np.ndarray:
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if np.dot(n, orig_normal) < 0:
                return tri[[0, 2, 1]]
            return tri

        quad_tri_0 = _align_winding(quad_tri_0)
        quad_tri_1 = _align_winding(quad_tri_1)
        alone_tri = _align_winding(alone_tri)

        if n_pos == 2:
            return [quad_tri_0, quad_tri_1], [alone_tri], seg_start, seg_end
        else:
            return [alone_tri], [quad_tri_0, quad_tri_1], seg_start, seg_end

    @staticmethod
    def cut(
        mesh: trimesh.Trimesh,
        plane: Plane,
    ) -> CutResult:
        dots = np.dot(mesh.vertices - plane.pos, plane.normal)

        lines, face_index = trimesh.intersections.mesh_plane(
            mesh, plane.normal, plane.pos, return_faces=True,
        )

        cut_face_set = set(face_index.tolist())

        pos_verts_list: List[np.ndarray] = []
        pos_faces_list: List[np.ndarray] = []
        neg_verts_list: List[np.ndarray] = []
        neg_faces_list: List[np.ndarray] = []
        pos_vert_count = 0
        neg_vert_count = 0

        # 处理未被切割的面: 按重心到平面的有符号距离分配
        for fi in range(len(mesh.faces)):
            if fi in cut_face_set:
                continue
            face = mesh.faces[fi]
            face_dots = dots[face]
            mean_dot = face_dots.mean()
            tri = mesh.vertices[face]
            if mean_dot >= 0:
                pos_faces_list.append(
                    np.array([[pos_vert_count, pos_vert_count + 1, pos_vert_count + 2]])
                )
                pos_verts_list.append(tri)
                pos_vert_count += 3
            else:
                neg_faces_list.append(
                    np.array([[neg_vert_count, neg_vert_count + 1, neg_vert_count + 2]])
                )
                neg_verts_list.append(tri)
                neg_vert_count += 3

        # 处理被切割的面
        for i, fi in enumerate(face_index):
            face = mesh.faces[fi]
            seg_start = lines[i, 0]
            seg_end = lines[i, 1]

            pos_tris, neg_tris, _, _ = PlaneMeshCutter._split_face(
                mesh.vertices, face, seg_start, seg_end, dots,
            )

            for tri in pos_tris:
                pos_faces_list.append(
                    np.array([[pos_vert_count, pos_vert_count + 1, pos_vert_count + 2]])
                )
                pos_verts_list.append(tri)
                pos_vert_count += 3

            for tri in neg_tris:
                neg_faces_list.append(
                    np.array([[neg_vert_count, neg_vert_count + 1, neg_vert_count + 2]])
                )
                neg_verts_list.append(tri)
                neg_vert_count += 3

        result = CutResult()

        # 边界环直接从 line segments 提取
        if len(lines) > 0:
            boundary_loops = PlaneMeshCutter._extract_boundary_loops_from_segments(lines)
        else:
            boundary_loops = []

        for tag, v_list, f_list in [
            ("positive", pos_verts_list, pos_faces_list),
            ("negative", neg_verts_list, neg_faces_list),
        ]:
            if len(v_list) == 0:
                result.meshes.append(None)
                result.boundary_loops.append([])
                continue

            all_verts = np.vstack(v_list)
            all_faces = np.vstack(f_list)
            part = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=True)
            result.meshes.append(part)
            result.boundary_loops.append(boundary_loops)

        if (
            len(result.meshes) == 2
            and result.meshes[0] is not None
            and result.meshes[1] is not None
            and len(result.boundary_loops[0]) > 0
            and len(result.boundary_loops[1]) > 0
        ):
            result.matched_loops = PlaneMeshCutter._match_boundary_loops(
                result.boundary_loops[0],
                result.boundary_loops[1],
            )

        result.meshes = [m for m in result.meshes if m is not None]

        return result

    @staticmethod
    def visualize(
        mesh_list: List[trimesh.Trimesh],
        plane: Union[Plane, None] = None,
    ) -> bool:
        if not mesh_list:
            print("[WARN][PlaneMeshCutter::visualize] empty mesh list")
            return False

        colors = [
            [0.8, 0.2, 0.2, 0.8],
            [0.2, 0.2, 0.8, 0.8],
            [0.2, 0.8, 0.2, 0.8],
            [0.8, 0.8, 0.2, 0.8],
            [0.8, 0.2, 0.8, 0.8],
            [0.2, 0.8, 0.8, 0.8],
        ]

        geometries = []
        for i, m in enumerate(mesh_list):
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(m.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(m.faces)
            o3d_mesh.compute_vertex_normals()

            color = colors[i % len(colors)][:3]
            o3d_mesh.paint_uniform_color(color)
            geometries.append(o3d_mesh)

        if plane is not None:
            all_vertices = np.vstack([m.vertices for m in mesh_list])
            center = all_vertices.mean(axis=0)
            extent = np.linalg.norm(all_vertices.max(axis=0) - all_vertices.min(axis=0))

            normal = plane.normal / np.linalg.norm(plane.normal)
            arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
            u = np.cross(normal, arbitrary)
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)

            half = extent * 0.6
            corners = np.array([
                plane.pos + half * (-u - v),
                plane.pos + half * (u - v),
                plane.pos + half * (u + v),
                plane.pos + half * (-u + v),
            ])
            plane_mesh = o3d.geometry.TriangleMesh()
            plane_mesh.vertices = o3d.utility.Vector3dVector(corners)
            plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
            plane_mesh.compute_vertex_normals()
            plane_mesh.paint_uniform_color([0.9, 0.9, 0.3])
            geometries.append(plane_mesh)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        geometries.append(coord_frame)

        o3d.visualization.draw_geometries(
            geometries,
            window_name="Plane Mesh Cut Result",
            width=1280,
            height=720,
        )
        return True
