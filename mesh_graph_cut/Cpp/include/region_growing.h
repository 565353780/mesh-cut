#pragma once

#include "nanoflann.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Point cloud adaptor for nanoflann
struct PointCloud {
  std::vector<std::array<double, 3>> points;

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return points.size(); }

  // Returns the dim'th component of the idx'th point in the class
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    return points[idx][dim];
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

/**
 * @brief 计算覆盖所有顶点的最小半径
 *
 * @param vertices 顶点坐标数组
 * @param seed_indices 种子点索引
 * @return double 最小覆盖半径
 */
const double
compute_min_radius_cover_all(const std::vector<std::array<double, 3>> &vertices,
                             const std::vector<size_t> &seed_indices);

/**
 * @brief 构建顶点到面片的映射
 *
 * @param faces 面片数组
 * @param num_vertices 顶点数量
 * @return std::vector<std::vector<size_t>> 顶点到面片的映射
 */
const std::vector<std::vector<size_t>>
build_vertex_to_face_map(const std::vector<std::array<size_t, 3>> &faces,
                         size_t num_vertices);

/**
 * @brief 查找与给定中心点在指定半径内连通的所有面片
 *
 * @param center_idx 中心点索引
 * @param vertices 顶点坐标数组
 * @param faces 面片数组
 * @param vertex_to_faces 顶点到面片的映射
 * @param radius 搜索半径
 * @return std::vector<size_t> 连通面片索引数组
 */
const std::vector<size_t> find_connected_faces(
    size_t center_idx, const std::vector<std::array<double, 3>> &vertices,
    const std::vector<std::array<size_t, 3>> &faces,
    const std::vector<std::vector<size_t>> &vertex_to_faces, double radius);

/**
 * @brief 并行区域生长算法
 *
 * @param vertices 顶点坐标数组
 * @param faces 面片数组
 * @param seed_indices 种子点索引
 * @param num_segments 分割数量
 * @return std::vector<std::vector<size_t>> 每个种子点对应的连通面片索引数组集合
 */
std::vector<std::vector<size_t>>
run_parallel_region_growing(const std::vector<std::array<double, 3>> &vertices,
                            const std::vector<std::array<size_t, 3>> &faces,
                            const std::vector<size_t> &seed_indices,
                            size_t num_segments);
