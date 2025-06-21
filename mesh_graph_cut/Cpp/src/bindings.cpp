#include "cut_mesh.h"
#include "region_growing.h"
#include "sample.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cut_cpp, m) {
  m.doc() = "C++ implementation of mesh graph cut algorithm"; // optional module
                                                              // docstring

  m.def("run_parallel_region_growing", &run_parallel_region_growing,
        "Run parallel region growing algorithm");

  m.def("compute_min_radius_cover_all", &compute_min_radius_cover_all,
        "region_growing.compute_min_radius_cover_all");

  m.def("farthest_point_sampling", &farthest_point_sampling,
        "sample.farthest_point_sampling");

  m.def("toSubMeshSamplePoints", &toSubMeshSamplePoints,
        "sample.toSubMeshSamplePoints");

  m.def("cutMesh", &cutMesh, "cut_mesh.cutMesh");
}
