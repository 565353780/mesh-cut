#include "region_growing.h"
#include <iostream>
#include <limits>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace mesh_graph_cut_cpp {

double
compute_min_radius_cover_all(const std::vector<std::array<double, 3>> &vertices,
                             const std::vector<size_t> &seed_indices) {
  // 构建点云数据
  PointCloud cloud;
  cloud.points.reserve(seed_indices.size());
  for (size_t idx : seed_indices) {
    cloud.points.push_back(vertices[idx]);
  }

  // 构建KD树
  using KDTreeType = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, PointCloud>, PointCloud, 3>;

  KDTreeType index(3, cloud, {10});
  index.buildIndex();

  double max_min_dist = 0.0;
  std::vector<uint32_t> ret_index(1);
  std::vector<double> out_dist_sqr(1);

  // 对每个顶点，找到最近的种子点
  for (size_t i = 0; i < vertices.size(); ++i) {
    const double query_pt[3] = {vertices[i][0], vertices[i][1], vertices[i][2]};
    index.knnSearch(query_pt, 1, ret_index.data(), out_dist_sqr.data());
    max_min_dist = std::max(max_min_dist, out_dist_sqr[0]);
  }

  return std::sqrt(max_min_dist);
}

std::vector<std::vector<size_t>>
build_vertex_to_face_map(const std::vector<std::array<size_t, 3>> &faces,
                         size_t num_vertices) {
  std::vector<std::vector<size_t>> vertex_to_faces(num_vertices);
  for (size_t i = 0; i < faces.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      vertex_to_faces[faces[i][j]].push_back(i);
    }
  }
  return vertex_to_faces;
}

// 辅助函数：计算点到点的距离平方
double distance_squared(const std::array<double, 3> &p1,
                        const std::array<double, 3> &p2) {
  double dx = p1[0] - p2[0];
  double dy = p1[1] - p2[1];
  double dz = p1[2] - p2[2];
  return dx * dx + dy * dy + dz * dz;
}

std::vector<size_t> find_connected_faces(
    size_t center_idx, const std::vector<std::array<double, 3>> &vertices,
    const std::vector<std::array<size_t, 3>> &faces,
    const std::vector<std::vector<size_t>> &vertex_to_faces, double radius) {
  double radius_squared = radius * radius;
  const auto &center = vertices[center_idx];

  // 找到球内的顶点
  std::unordered_set<size_t> vertices_in_ball;
  for (size_t i = 0; i < vertices.size(); ++i) {
    if (distance_squared(vertices[i], center) <= radius_squared) {
      vertices_in_ball.insert(i);
    }
  }

  std::unordered_set<size_t> connected_faces;
  std::unordered_set<size_t> visited_vertices;
  std::queue<size_t> vertex_queue;

  // 从中心点开始BFS
  vertex_queue.push(center_idx);

  while (!vertex_queue.empty()) {
    size_t current_vertex = vertex_queue.front();
    vertex_queue.pop();

    if (visited_vertices.count(current_vertex) > 0 ||
        vertices_in_ball.count(current_vertex) == 0) {
      continue;
    }

    visited_vertices.insert(current_vertex);

    // 检查与当前顶点相邻的所有面片
    for (size_t face_id : vertex_to_faces[current_vertex]) {
      const auto &face = faces[face_id];

      // 检查面片的所有顶点是否都在球内
      bool all_vertices_in_ball = true;
      for (size_t vertex_idx : face) {
        if (vertices_in_ball.count(vertex_idx) == 0) {
          all_vertices_in_ball = false;
          break;
        }
      }

      // 如果面片所有顶点都在球内，将其加入结果，并将未访问的顶点加入队列
      if (all_vertices_in_ball) {
        connected_faces.insert(face_id);
        for (size_t vertex_idx : face) {
          if (visited_vertices.count(vertex_idx) == 0) {
            vertex_queue.push(vertex_idx);
          }
        }
      }
    }
  }

  return std::vector<size_t>(connected_faces.begin(), connected_faces.end());
}

std::vector<size_t>
run_parallel_region_growing(const std::vector<std::array<double, 3>> &vertices,
                            const std::vector<std::array<size_t, 3>> &faces,
                            const std::vector<size_t> &seed_indices,
                            size_t num_segments) {
  // 计算覆盖半径
  double radius = compute_min_radius_cover_all(vertices, seed_indices);

  // 构建顶点到面片的映射
  auto vertex_to_faces = build_vertex_to_face_map(faces, vertices.size());

  // 初始化结果数组，-1表示未分配
  std::vector<size_t> face_labels(faces.size(),
                                  std::numeric_limits<size_t>::max());

  // 对每个种子点进行区域生长
  for (size_t i = 0; i < seed_indices.size(); ++i) {
    auto connected_faces = find_connected_faces(seed_indices[i], vertices,
                                                faces, vertex_to_faces, radius);

    // 标记连通的面片
    for (size_t face_id : connected_faces) {
      // 如果面片未被标记或当前种子点更近，则更新标签
      if (face_labels[face_id] == std::numeric_limits<size_t>::max()) {
        face_labels[face_id] = i;
      }
    }
  }

  return face_labels;
}

} // namespace mesh_graph_cut_cpp
