#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "region_growing.h"
#include "sample.h"

namespace py = pybind11;

// 前向声明
py::array_t<float>
toSubMeshSamplePoints(py::array_t<float> vertices, py::array_t<int> triangles,
                      const std::vector<std::vector<size_t>> &face_groups,
                      const int &points_per_submesh);

PYBIND11_MODULE(mesh_graph_cut_cpp, m) {
  m.doc() = "C++ implementation of mesh graph cut algorithm"; // optional module
                                                              // docstring

  // 导出区域生长算法
  m.def("run_parallel_region_growing", &run_parallel_region_growing,
        "Run parallel region growing algorithm", py::arg("vertices"),
        py::arg("faces"), py::arg("seed_indices"), py::arg("num_segments"));

  // 导出计算最小覆盖半径的函数
  m.def("compute_min_radius_cover_all", &compute_min_radius_cover_all,
        "Compute minimum radius to cover all vertices", py::arg("vertices"),
        py::arg("seed_indices"));

  // 导出最远点采样算法
  m.def("farthest_point_sampling", &farthest_point_sampling,
        "Perform farthest point sampling on a point cloud", py::arg("points"),
        py::arg("sample_point_num"));

  // 导出子网格均匀采样函数
  m.def("toSubMeshSamplePoints", &toSubMeshSamplePoints,
        "Uniformly sample points from submeshes based on face groups",
        py::arg("vertices"), py::arg("triangles"), py::arg("face_groups"),
        py::arg("points_per_submesh"),
        "Performs uniform sampling on each submesh (specified by face groups) "
        "to generate points.\n"
        "Args:\n"
        "    vertices: (N, 3) float array of vertex coordinates\n"
        "    triangles: (M, 3) int array of triangle vertex indices\n"
        "    face_groups: list of lists, where each inner list contains face "
        "indices for a submesh\n"
        "Returns:\n"
        "    (K, 8192, 3) float array of sampled points for each submesh, "
        "where K is the number of face groups");
}
