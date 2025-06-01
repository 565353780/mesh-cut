#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <limits>
#include <algorithm>
#include <functional>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <memory>
#include "kdtree.h"
#include "halfedge.h"

namespace mesh_graph_cut_cpp {

/**
 * @brief 计算覆盖所有顶点的最小半径
 * 
 * @param vertices 顶点坐标数组
 * @param seed_indices 种子点索引
 * @return double 最小覆盖半径
 */
double compute_min_radius_cover_all(
    const std::vector<std::array<double, 3>>& vertices,
    const std::vector<size_t>& seed_indices);

/**
 * @brief 构建顶点到面片的映射
 * 
 * @param faces 面片数组
 * @param num_vertices 顶点数量
 * @return std::vector<std::vector<size_t>> 顶点到面片的映射
 */
std::vector<std::vector<size_t>> build_vertex_to_face_map(
    const std::vector<std::array<size_t, 3>>& faces,
    size_t num_vertices);

/**
 * @brief 查找与给定面片连通的所有面片
 * 
 * @param start_face 起始面片索引
 * @param faces 面片数组
 * @param vertex_to_faces 顶点到面片的映射
 * @param max_curvature 最大曲率阈值
 * @param face_curvatures 面片曲率数组
 * @param visited 已访问的面片集合
 * @return std::vector<size_t> 连通面片索引数组
 */
std::vector<size_t> find_connected_faces(
    size_t start_face,
    const std::vector<std::array<size_t, 3>>& faces,
    const std::vector<std::vector<size_t>>& vertex_to_faces,
    double max_curvature,
    const std::vector<double>& face_curvatures,
    std::vector<bool>& visited);

/**
 * @brief 并行区域生长算法
 * 
 * @param vertices 顶点坐标数组
 * @param faces 面片数组
 * @param vertex_curvatures 顶点曲率数组
 * @param face_curvatures 面片曲率数组
 * @param seed_indices 种子点索引
 * @param num_segments 分割数量
 * @return std::vector<size_t> 面片标签数组
 */
std::vector<size_t> run_parallel_region_growing(
    const std::vector<std::array<double, 3>>& vertices,
    const std::vector<std::array<size_t, 3>>& faces,
    const std::vector<double>& vertex_curvatures,
    const std::vector<double>& face_curvatures,
    const std::vector<size_t>& seed_indices,
    size_t num_segments);

} // namespace mesh_graph_cut_cpp