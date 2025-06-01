#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "region_growing.h"

namespace py = pybind11;

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
}
