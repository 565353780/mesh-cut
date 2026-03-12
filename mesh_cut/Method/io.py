import os
import trimesh

from typing import Optional, Union


def postProcessMesh(mesh: Union[trimesh.Trimesh, trimesh.Scene],
) -> Optional[trimesh.Trimesh]:
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    if not isinstance(mesh, trimesh.Trimesh):
        print('[ERROR][io::postProcessMesh]')
        print(f'\t Loaded object is not a Trimesh, got type: {type(mesh)}')
        return None

    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
        mesh.vertex_normals = mesh.vertex_normals

    return mesh

def loadMeshFile(
    mesh_file_path: str,
) -> Optional[trimesh.Trimesh]:
    if not os.path.exists(mesh_file_path):
        print('[ERROR][io::loadMeshFile]')
        print('\t mesh file not exist!')
        print('\t mesh_file_path:', mesh_file_path)
        return None

    mesh = trimesh.load(mesh_file_path, process=False)

    mesh = postProcessMesh(mesh)
    if mesh is None:
        print('[ERROR][io::loadMeshFile]')
        print('\t postProcessMesh failed!')

    return mesh
