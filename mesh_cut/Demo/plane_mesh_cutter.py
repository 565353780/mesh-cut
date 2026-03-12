import os

from mesh_cut.Data.plane import Plane
from mesh_cut.Method.io import loadMeshFile
from mesh_cut.Module.plane_mesh_cutter import PlaneMeshCutter


def demo():
    mesh_file_path = "/Users/chli/Downloads/tmp/mesh_cut_results/Hitem3d-1773294165381.glb"
    output_dir = "/Users/chli/Downloads/tmp/mesh_cut_results/"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading mesh from", mesh_file_path)
    mesh = loadMeshFile(mesh_file_path)
    if mesh is None:
        print("[ERROR][demo] failed to load mesh")
        return False

    plane = Plane(
        pos=mesh.centroid,
        normal=[0, 1, 0],
    )

    print("Cutting mesh by plane...")
    cut_result = PlaneMeshCutter.cut(mesh, plane)

    print(f"Cut into {len(cut_result.meshes)} parts")
    for i, m in enumerate(cut_result.meshes):
        print(f"  Part {i}: {len(m.vertices)} vertices, {len(m.faces)} faces")
        m.export(output_dir + f'{i:06d}.ply')

    print(f"{len(cut_result.boundary_loops)} boundary loops")
    for j, loop in enumerate(cut_result.boundary_loops):
        print(f"  loop {j}: {len(loop)} vertex pairs")

    #PlaneMeshCutter.visualize(cut_result.meshes, plane)
    return True
