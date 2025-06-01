#include "halfedge.h"
#include "kdtree.h"
#include "region_growing.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace mesh_graph_cut_cpp {

// 将NumPy数组转换为C++数据结构
std::vector<std::array<double, 3>>
convert_vertices(py::array_t<double> vertices_array) {
  auto vertices_buffer = vertices_array.request();
  if (vertices_buffer.ndim != 2 || vertices_buffer.shape[1] != 3) {
    throw std::runtime_error("Vertices array must be of shape (n, 3)");
  }

  size_t num_vertices = vertices_buffer.shape[0];
  double *vertices_ptr = static_cast<double *>(vertices_buffer.ptr);

  std::vector<std::array<double, 3>> vertices(num_vertices);
  for (size_t i = 0; i < num_vertices; ++i) {
    vertices[i][0] = vertices_ptr[i * 3];
    vertices[i][1] = vertices_ptr[i * 3 + 1];
    vertices[i][2] = vertices_ptr[i * 3 + 2];
  }

  return vertices;
}

std::vector<std::array<size_t, 3>>
convert_faces(py::array_t<size_t> faces_array) {
  auto faces_buffer = faces_array.request();
  if (faces_buffer.ndim != 2 || faces_buffer.shape[1] != 3) {
    throw std::runtime_error("Faces array must be of shape (n, 3)");
  }

  size_t num_faces = faces_buffer.shape[0];
  size_t *faces_ptr = static_cast<size_t *>(faces_buffer.ptr);

  std::vector<std::array<size_t, 3>> faces(num_faces);
  for (size_t i = 0; i < num_faces; ++i) {
    faces[i][0] = faces_ptr[i * 3];
    faces[i][1] = faces_ptr[i * 3 + 1];
    faces[i][2] = faces_ptr[i * 3 + 2];
  }

  return faces;
}

std::vector<double> convert_curvatures(py::array_t<double> curvatures_array) {
  auto curvatures_buffer = curvatures_array.request();
  if (curvatures_buffer.ndim != 1) {
    throw std::runtime_error("Curvatures array must be 1-dimensional");
  }

  size_t num_curvatures = curvatures_buffer.shape[0];
  double *curvatures_ptr = static_cast<double *>(curvatures_buffer.ptr);

  std::vector<double> curvatures(num_curvatures);
  for (size_t i = 0; i < num_curvatures; ++i) {
    curvatures[i] = curvatures_ptr[i];
  }

  return curvatures;
}

std::vector<size_t> convert_indices(py::array_t<size_t> indices_array) {
  auto indices_buffer = indices_array.request();
  if (indices_buffer.ndim != 1) {
    throw std::runtime_error("Indices array must be 1-dimensional");
  }

  size_t num_indices = indices_buffer.shape[0];
  size_t *indices_ptr = static_cast<size_t *>(indices_buffer.ptr);

  std::vector<size_t> indices(num_indices);
  for (size_t i = 0; i < num_indices; ++i) {
    indices[i] = indices_ptr[i];
  }

  return indices;
}

// 包装函数，用于从Python调用
py::array_t<size_t> py_run_parallel_region_growing(
    py::array_t<double> vertices_array, py::array_t<size_t> faces_array,
    py::array_t<double> vertex_curvatures_array,
    py::array_t<double> face_curvatures_array,
    py::array_t<size_t> seed_indices_array, size_t num_segments) {

  // 转换输入数据
  std::vector<std::array<double, 3>> vertices =
      convert_vertices(vertices_array);
  std::vector<std::array<size_t, 3>> faces = convert_faces(faces_array);
  std::vector<double> vertex_curvatures =
      convert_curvatures(vertex_curvatures_array);
  std::vector<double> face_curvatures =
      convert_curvatures(face_curvatures_array);
  std::vector<size_t> seed_indices = convert_indices(seed_indices_array);

  // 调用C++实现
  std::vector<size_t> face_labels =
      run_parallel_region_growing(vertices, faces, vertex_curvatures,
                                  face_curvatures, seed_indices, num_segments);

  // 创建NumPy数组返回结果
  py::array_t<size_t> result = py::array_t<size_t>(face_labels.size());
  auto result_buffer = result.request();
  size_t *result_ptr = static_cast<size_t *>(result_buffer.ptr);

  for (size_t i = 0; i < face_labels.size(); ++i) {
    result_ptr[i] = face_labels[i];
  }

  return result;
}

py::array_t<size_t>
py_find_connected_faces(size_t start_face, py::array_t<size_t> faces_array,
                        py::array_t<double> face_curvatures_array,
                        double max_curvature, size_t num_vertices) {

  // 转换输入数据
  std::vector<std::array<size_t, 3>> faces = convert_faces(faces_array);
  std::vector<double> face_curvatures =
      convert_curvatures(face_curvatures_array);

  // 构建顶点到面片的映射
  std::vector<std::vector<size_t>> vertex_to_faces =
      build_vertex_to_face_map(faces, num_vertices);

  // 初始化访问标记
  std::vector<bool> visited(faces.size(), false);

  // 调用C++实现
  std::vector<size_t> connected_faces =
      find_connected_faces(start_face, faces, vertex_to_faces, max_curvature,
                           face_curvatures, visited);

  // 创建NumPy数组返回结果
  py::array_t<size_t> result = py::array_t<size_t>(connected_faces.size());
  auto result_buffer = result.request();
  size_t *result_ptr = static_cast<size_t *>(result_buffer.ptr);

  for (size_t i = 0; i < connected_faces.size(); ++i) {
    result_ptr[i] = connected_faces[i];
  }

  return result;
}

py::array_t<double>
py_compute_min_radius_cover_all(py::array_t<double> vertices_array,
                                py::array_t<size_t> seed_indices_array) {

  // 转换输入数据
  std::vector<std::array<double, 3>> vertices =
      convert_vertices(vertices_array);
  std::vector<size_t> seed_indices = convert_indices(seed_indices_array);

  // 调用C++实现
  double radius = compute_min_radius_cover_all(vertices, seed_indices);

  // 创建NumPy数组返回结果
  py::array_t<double> result = py::array_t<double>(1);
  auto result_buffer = result.request();
  double *result_ptr = static_cast<double *>(result_buffer.ptr);
  result_ptr[0] = radius;

  return result;
}

py::list py_build_vertex_to_face_map(py::array_t<size_t> faces_array,
                                     size_t num_vertices) {

  // 转换输入数据
  std::vector<std::array<size_t, 3>> faces = convert_faces(faces_array);

  // 调用C++实现
  std::vector<std::vector<size_t>> vertex_to_faces =
      build_vertex_to_face_map(faces, num_vertices);

  // 创建Python列表返回结果
  py::list result;
  for (const auto &face_list : vertex_to_faces) {
    py::list py_face_list;
    for (size_t face_idx : face_list) {
      py_face_list.append(face_idx);
    }
    result.append(py_face_list);
  }

  return result;
}

// 包装HalfEdgeStructure类
class PyHalfEdgeStructure {
public:
  PyHalfEdgeStructure(py::array_t<size_t> faces_array, size_t num_vertices) {
    std::vector<std::array<size_t, 3>> faces = convert_faces(faces_array);
    halfedge_ = std::make_unique<HalfEdgeStructure>(faces, num_vertices);
  }

  py::list get_vertex_neighbors(size_t vertex_index) const {
    std::vector<size_t> neighbors =
        halfedge_->get_vertex_neighbors(vertex_index);
    py::list result;
    for (size_t neighbor : neighbors) {
      result.append(neighbor);
    }
    return result;
  }

  py::list get_vertex_faces(size_t vertex_index) const {
    std::vector<size_t> faces = halfedge_->get_vertex_faces(vertex_index);
    py::list result;
    for (size_t face : faces) {
      result.append(face);
    }
    return result;
  }

  py::list get_face_neighbors(size_t face_index) const {
    std::vector<size_t> neighbors = halfedge_->get_face_neighbors(face_index);
    py::list result;
    for (size_t neighbor : neighbors) {
      result.append(neighbor);
    }
    return result;
  }

private:
  std::unique_ptr<HalfEdgeStructure> halfedge_;
};

} // namespace mesh_graph_cut_cpp

PYBIND11_MODULE(mesh_graph_cut_cpp, m) {
  m.doc() = "C++ accelerated mesh graph cutting algorithms";

  // 导出函数
  m.def("run_parallel_region_growing",
        &mesh_graph_cut_cpp::py_run_parallel_region_growing,
        "Run parallel region growing algorithm", py::arg("vertices"),
        py::arg("faces"), py::arg("vertex_curvatures"),
        py::arg("face_curvatures"), py::arg("seed_indices"),
        py::arg("num_segments"));

  m.def("find_connected_faces", &mesh_graph_cut_cpp::py_find_connected_faces,
        "Find connected faces with curvature below threshold",
        py::arg("start_face"), py::arg("faces"), py::arg("face_curvatures"),
        py::arg("max_curvature"), py::arg("num_vertices"));

  m.def("compute_min_radius_cover_all",
        &mesh_graph_cut_cpp::py_compute_min_radius_cover_all,
        "Compute minimum radius to cover all vertices", py::arg("vertices"),
        py::arg("seed_indices"));

  m.def("build_vertex_to_face_map",
        &mesh_graph_cut_cpp::py_build_vertex_to_face_map,
        "Build mapping from vertices to faces", py::arg("faces"),
        py::arg("num_vertices"));

  // 导出HalfEdgeStructure类
  py::class_<mesh_graph_cut_cpp::PyHalfEdgeStructure>(m, "HalfEdgeStructure")
      .def(py::init<py::array_t<size_t>, size_t>(), py::arg("faces"),
           py::arg("num_vertices"))
      .def("get_vertex_neighbors",
           &mesh_graph_cut_cpp::PyHalfEdgeStructure::get_vertex_neighbors,
           "Get neighboring vertices of a vertex", py::arg("vertex_index"))
      .def("get_vertex_faces",
           &mesh_graph_cut_cpp::PyHalfEdgeStructure::get_vertex_faces,
           "Get faces containing a vertex", py::arg("vertex_index"))
      .def("get_face_neighbors",
           &mesh_graph_cut_cpp::PyHalfEdgeStructure::get_face_neighbors,
           "Get neighboring faces of a face", py::arg("face_index"));
}
