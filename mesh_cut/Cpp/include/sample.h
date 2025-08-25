#pragma once

#include <random>
#include <torch/extension.h>
#include <vector>

// 最远点采样算法的C++实现
torch::Tensor farthest_point_sampling(torch::Tensor points,
                                      int sample_point_num);

/**
 * @brief 对N个子网格进行并行采样，每个子网格采样M个点
 *
 * @param vertices 顶点坐标张量，形状为(N, 3)的torch::Tensor
 * @param triangles 三角形面片张量，形状为(M, 3)的torch::Tensor
 * @param face_groups 面片组数组，每个元素是一个包含面片索引的数组
 * @param points_per_submesh 每个子网格采样的点数
 * @return torch::Tensor 采样点张量，形状为(4000, 8192, 3)的torch::Tensor
 */
torch::Tensor
toSubMeshSamplePoints(torch::Tensor vertices, torch::Tensor triangles,
                      const std::vector<std::vector<size_t>> &face_groups,
                      const int &points_per_submesh);
