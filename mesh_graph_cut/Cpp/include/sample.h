#pragma once

#include <pybind11/numpy.h>
#include <random>
#include <vector>

namespace py = pybind11;

// 最远点采样算法的C++实现
std::vector<int> farthest_point_sampling(py::array_t<float> points,
                                         int sample_point_num);

/**
 * @brief 对N个子网格进行并行采样，每个子网格采样M个点
 * 
 * @param vertices 顶点坐标数组，形状为(N, 3)的numpy数组
 * @param triangles 三角形面片数组，形状为(M, 3)的numpy数组
 * @param face_groups 面片组数组，每个元素是一个包含面片索引的数组
 * @return py::array_t<float> 采样点数组，形状为(4000, 8192, 3)的numpy数组
 */
py::array_t<float> toSubMeshSamplePoints(py::array_t<float> vertices,
                                         py::array_t<int> triangles,
                                         const std::vector<std::vector<size_t>>& face_groups);