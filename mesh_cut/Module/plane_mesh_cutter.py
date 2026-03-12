import trimesh
import numpy as np
import open3d as o3d

from typing import List, Union
from mesh_cut.Data.plane import Plane


class PlaneMeshCutter(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def cut(
        mesh: trimesh.Trimesh,
        plane: Plane,
    ) -> List[trimesh.Trimesh]:
        positive_part = trimesh.intersections.slice_mesh_plane(
            mesh, plane.normal, plane.pos, cap=False,
        )

        negative_part = trimesh.intersections.slice_mesh_plane(
            mesh, -plane.normal, plane.pos, cap=False,
        )

        result = []
        if positive_part is not None and len(positive_part.faces) > 0:
            result.append(positive_part)
        if negative_part is not None and len(negative_part.faces) > 0:
            result.append(negative_part)

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
