#include "region_growing.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>

namespace mesh_graph_cut_cpp {

std::vector<std::vector<size_t>>
build_vertex_to_face_map(const std::vector<std::array<size_t, 3>> &faces,
                         size_t num_vertices) {

  // 初始化顶点到面片的映射
  std::vector<std::vector<size_t>> vertex_to_faces(num_vertices);

  // 遍历所有面片
  for (size_t face_idx = 0; face_idx < faces.size(); ++face_idx) {
    const auto &face = faces[face_idx];
    // 对于面片中的每个顶点
    for (size_t i = 0; i < 3; ++i) {
      // 将面片索引添加到对应顶点的列表中
      vertex_to_faces[face[i]].push_back(face_idx);
    }
  }

  return vertex_to_faces;
}

double
compute_min_radius_cover_all(const std::vector<std::array<double, 3>> &vertices,
                             const std::vector<size_t> &seed_indices) {

  if (seed_indices.empty() || vertices.empty()) {
    return 0.0;
  }

  // 提取种子点坐标
  std::vector<std::array<double, 3>> seed_points;
  seed_points.reserve(seed_indices.size());
  for (size_t idx : seed_indices) {
    seed_points.push_back(vertices[idx]);
  }

  // 创建种子点的KD树
  KDTree seed_kdtree(seed_points);

  // 计算每个顶点到最近种子点的距离
  std::vector<double> min_distances(vertices.size());

  // 并行处理顶点批次
  const size_t batch_size = 1000; // 每批处理的顶点数量
  const size_t num_vertices = vertices.size();
  const size_t num_batches = (num_vertices + batch_size - 1) / batch_size;

  std::vector<std::future<void>> futures;

  for (size_t batch = 0; batch < num_batches; ++batch) {
    size_t start_idx = batch * batch_size;
    size_t end_idx = std::min(start_idx + batch_size, num_vertices);

    futures.push_back(std::async(std::launch::async, [&, start_idx, end_idx]() {
      // 为批次中的每个顶点找到最近的种子点
      for (size_t i = start_idx; i < end_idx; ++i) {
        // 使用KD树查找最近的种子点
        size_t nearest_seed_idx = seed_kdtree.nearest(vertices[i]);

        // 计算距离
        min_distances[i] =
            KDTree::distance(vertices[i], seed_points[nearest_seed_idx]);
      }
    }));
  }

  // 等待所有任务完成
  for (auto &future : futures) {
    future.wait();
  }

  // 找到最大的最小距离，即覆盖半径
  double max_min_distance = 0.0;
  for (double dist : min_distances) {
    max_min_distance = std::max(max_min_distance, dist);
  }

  return max_min_distance;
}

std::vector<size_t>
find_connected_faces(size_t start_face,
                     const std::vector<std::array<size_t, 3>> &faces,
                     const std::vector<std::vector<size_t>> &vertex_to_faces,
                     std::vector<bool> &visited) {

  std::vector<size_t> connected_faces;
  std::queue<size_t> queue;

  // 标记起始面片为已访问
  visited[start_face] = true;
  queue.push(start_face);
  connected_faces.push_back(start_face);

  // 使用BFS查找连通区域，只考虑拓扑连通性
  while (!queue.empty()) {
    size_t current_face = queue.front();
    queue.pop();

    // 获取当前面片的顶点
    const auto &face = faces[current_face];

    // 对于面片的每个顶点
    for (size_t i = 0; i < 3; ++i) {
      size_t vertex = face[i];

      // 获取包含该顶点的所有面片（共享顶点的面片）
      for (size_t neighbor_face : vertex_to_faces[vertex]) {
        // 只检查是否已访问，不再考虑曲率
        if (!visited[neighbor_face]) {
          visited[neighbor_face] = true;
          queue.push(neighbor_face);
          connected_faces.push_back(neighbor_face);
        }
      }
    }
  }

  return connected_faces;
}

std::vector<size_t>
run_parallel_region_growing(const std::vector<std::array<double, 3>> &vertices,
                            const std::vector<std::array<size_t, 3>> &faces,
                            const std::vector<size_t> &seed_indices,
                            size_t num_segments) {

  auto start_time = std::chrono::high_resolution_clock::now();
  std::cout << "Starting parallel region growing with " << num_segments
            << " segments" << std::endl;

  // 构建KD树用于空间查询
  std::vector<std::array<double, 3>> seed_points;
  seed_points.reserve(seed_indices.size());
  for (size_t idx : seed_indices) {
    seed_points.push_back(vertices[idx]);
  }
  KDTree seed_kdtree(seed_points);

  // 构建顶点到面片的映射
  std::cout << "Building vertex to face map..." << std::endl;
  std::vector<std::vector<size_t>> vertex_to_faces =
      build_vertex_to_face_map(faces, vertices.size());

  // 初始化结果数组，-1表示未分配
  std::vector<size_t> face_labels(faces.size(), static_cast<size_t>(-1));

  // 计算覆盖半径
  std::cout << "Computing coverage radius..." << std::endl;
  double radius = compute_min_radius_cover_all(vertices, seed_indices);
  std::cout << "Coverage radius: " << radius << std::endl;

  // 初始化访问标记
  std::vector<bool> visited(faces.size(), false);

  // 并行处理每个种子点
  std::vector<std::vector<size_t>> segments(seed_indices.size());
  std::vector<std::future<void>> futures;

  std::cout << "Growing regions from seeds..." << std::endl;
  for (size_t i = 0; i < std::min(seed_indices.size(), num_segments); ++i) {
    futures.push_back(std::async(std::launch::async, [&, i]() {
      const auto &seed_point = vertices[seed_indices[i]];
      std::vector<bool> local_visited = visited; // 局部副本

      // 找到种子点附近的所有面片
      for (size_t face_idx = 0; face_idx < faces.size(); ++face_idx) {
        if (local_visited[face_idx])
          continue;

        // 计算面片中心
        const auto &face = faces[face_idx];
        std::array<double, 3> face_center = {0, 0, 0};
        for (size_t j = 0; j < 3; ++j) {
          for (size_t k = 0; k < 3; ++k) {
            face_center[k] += vertices[face[j]][k] / 3.0;
          }
        }

        // 检查面片中心是否在种子点的覆盖半径内
        double dist = KDTree::distance(face_center, seed_point);
        if (dist <= radius) {
          // 查找连通区域，不再使用曲率阈值
          auto connected = find_connected_faces(face_idx, faces,
                                                vertex_to_faces, local_visited);

          // 将连通区域添加到当前分段
          segments[i].insert(segments[i].end(), connected.begin(),
                             connected.end());
        }
      }
    }));
  }

  // 等待所有任务完成
  for (auto &future : futures) {
    future.wait();
  }

  // 合并结果，使用最近种子点的标签
  std::cout << "Merging results..." << std::endl;
  for (size_t face_idx = 0; face_idx < faces.size(); ++face_idx) {
    if (face_labels[face_idx] == static_cast<size_t>(-1)) {
      // 计算面片中心
      std::array<double, 3> face_center = {0, 0, 0};
      const auto &face = faces[face_idx];
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < 3; ++k) {
          face_center[k] += vertices[face[j]][k];
        }
      }
      for (size_t k = 0; k < 3; ++k) {
        face_center[k] /= 3.0;
      }

      // 找到最近的种子点
      size_t nearest_seed = seed_kdtree.nearest(face_center);
      // 确保标签在有效范围内
      if (nearest_seed < std::min(seed_indices.size(), num_segments)) {
        face_labels[face_idx] = nearest_seed;
      }
    }
  }

  // 计算执行时间
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_time - start_time)
                      .count();
  std::cout << "Parallel region growing completed in " << duration << " ms"
            << std::endl;

  return face_labels;
}

} // namespace mesh_graph_cut_cpp
