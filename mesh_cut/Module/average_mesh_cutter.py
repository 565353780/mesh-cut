import torch
from typing import Union

from cut_cpp import (
    farthest_point_sampling,
    run_parallel_region_growing,
    toSubMeshSamplePoints,
)

from mesh_cut.Module.base_mesh_cutter import BaseMeshCutter


class AverageMeshCutter(BaseMeshCutter):
    def __init__(
        self,
        mesh_file_path: Union[str, None] = None,
        dist_max: float = 1.0 / 500,
    ):
        super().__init__(mesh_file_path, dist_max)
        return

    def cutMesh(
        self,
        sub_mesh_num: int = 400,
        points_per_submesh: int = 8192,
    ) -> Union[list, bool]:
        self.subdivMesh(10 * sub_mesh_num)

        if not self.isValid():
            print("[ERROR][AverageMeshCutter::cutMesh]")
            print("\t mesh is not valid!")
            return False

        self.center_vertex_idxs = farthest_point_sampling(
            torch.from_numpy(self.vertices).to(torch.float32), sub_mesh_num
        )

        self.face_labels = run_parallel_region_growing(
            self.vertices,
            self.triangles,
            self.center_vertex_idxs.numpy(),
            sub_mesh_num,
        )

        self.sub_mesh_sample_points = toSubMeshSamplePoints(
            torch.from_numpy(self.vertices).to(torch.float32),
            torch.from_numpy(self.triangles).to(torch.int),
            self.face_labels,
            points_per_submesh,
        )
        return True
