#pragma once

#include <vector>

void cutMesh(const std::vector<float> &vertices,
             const std::vector<int> &triangles,
             const std::vector<float> &cut_vertices,
             const std::vector<int> &cut_triangles);
