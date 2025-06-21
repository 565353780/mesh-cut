#pragma once

#include <torch/extension.h>

void cutMesh(const torch::Tensor &vertices, const torch::Tensor &triangles,
             const torch::Tensor &cut_vertices,
             const torch::Tensor &cut_triangles);
