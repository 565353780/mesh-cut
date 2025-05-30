from mesh_graph_cut.Module.mesh_graph_cutter import MeshGraphCutter


def demo():
    mesh_file_path = "/Users/chli/Downloads/Dataset/vae测评/000.obj"
    sub_mesh_num = 400

    mesh_graph_cutter = MeshGraphCutter(mesh_file_path)

    mesh_graph_cutter.cutMesh(sub_mesh_num)
    return True
