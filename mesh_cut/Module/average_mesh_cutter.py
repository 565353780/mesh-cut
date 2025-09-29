import torch
import numpy as np
from math import ceil
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
        is_unique_label: bool = False,
        print_progress: bool = True,
    ):
        self.is_unique_label = is_unique_label

        super().__init__(mesh_file_path, dist_max, print_progress)
        return

    def cutMesh(
        self,
        sub_mesh_num: int = 400,
        points_per_submesh: int = 8192,
        subdiv_scale: float = 10.0,
    ) -> Union[list, bool]:
        if subdiv_scale > 1.0:
            self.subdivMesh(ceil(subdiv_scale * sub_mesh_num))

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

        if self.is_unique_label:
            unique_face_labels = np.zeros(self.triangles.shape[0]).astype(np.int32) * -1

            for i, submesh_faces in enumerate(self.face_labels):
                unique_face_labels[submesh_faces] = i

            self.face_labels = unique_face_labels

        if points_per_submesh > 0:
            self.sub_mesh_sample_points = toSubMeshSamplePoints(
                torch.from_numpy(self.vertices).to(torch.float32),
                torch.from_numpy(self.triangles).to(torch.int),
                self.face_labels.reshape(-1, 1),
                points_per_submesh,
            )
        return True
