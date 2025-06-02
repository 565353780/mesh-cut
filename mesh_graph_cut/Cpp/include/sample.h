#pragma once

#include <pybind11/numpy.h>
#include <random>
#include <vector>

namespace py = pybind11;

// 最远点采样算法的C++实现
std::vector<int> farthest_point_sampling(py::array_t<float> points,
                                         int sample_point_num);
