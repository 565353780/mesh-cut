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
    for i, (m, loops) in enumerate(zip(cut_result.meshes, cut_result.boundary_loops)):
        total_boundary_verts = sum(len(lp) for lp in loops)
        print(f"  Part {i}: {len(m.vertices)} vertices, {len(m.faces)} faces, "
              f"{len(loops)} boundary loops, {total_boundary_verts} boundary vertices")
        m.export(output_dir + f'{i:06d}.ply')

    if cut_result.matched_loops:
        print(f"Matched {len(cut_result.matched_loops)} boundary loop pairs:")
        for match in cut_result.matched_loops:
            pos_loop = cut_result.boundary_loops[0][match.positive_loop_idx]
            neg_loop = cut_result.boundary_loops[1][match.negative_loop_idx]
            print(f"  positive loop {match.positive_loop_idx} ({len(pos_loop)} verts) "
                  f"<-> negative loop {match.negative_loop_idx} ({len(neg_loop)} verts), "
                  f"cost={match.transport_cost:.6f}")

    #PlaneMeshCutter.visualize(cut_result.meshes, plane)
    return True
