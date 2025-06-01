#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mesh_graph_cut_cpp {

/**
 * @brief 半边结构中的半边
 */
struct HalfEdge {
  size_t origin;      // 起点索引
  size_t destination; // 终点索引
  size_t face;        // 所属面片索引
  size_t next;        // 下一条半边索引
  size_t twin;        // 对偶半边索引，如果没有对偶边则为-1

  HalfEdge(size_t o, size_t d, size_t f, size_t n,
           size_t t = static_cast<size_t>(-1))
      : origin(o), destination(d), face(f), next(n), twin(t) {}
};

/**
 * @brief 半边结构，用于加速网格拓扑查询
 */
class HalfEdgeStructure {
public:
  /**
   * @brief 构造函数
   * @param faces 面片数组
   * @param num_vertices 顶点数量
   */
  HalfEdgeStructure(const std::vector<std::array<size_t, 3>> &faces,
                    size_t num_vertices);

  /**
   * @brief 获取顶点的所有相邻顶点
   * @param vertex_index 顶点索引
   * @return 相邻顶点索引数组
   */
  std::vector<size_t> get_vertex_neighbors(size_t vertex_index) const;

  /**
   * @brief 获取顶点的所有相邻面片
   * @param vertex_index 顶点索引
   * @return 相邻面片索引数组
   */
  std::vector<size_t> get_vertex_faces(size_t vertex_index) const;

  /**
   * @brief 获取面片的相邻面片
   * @param face_index 面片索引
   * @return 相邻面片索引数组
   */
  std::vector<size_t> get_face_neighbors(size_t face_index) const;

  /**
   * @brief 获取从顶点出发的半边
   * @param vertex_index 顶点索引
   * @return 半边索引数组
   */
  std::vector<size_t> get_outgoing_halfedges(size_t vertex_index) const;

  /**
   * @brief 获取面片的半边
   * @param face_index 面片索引
   * @return 半边索引数组
   */
  std::array<size_t, 3> get_face_halfedges(size_t face_index) const;

  /**
   * @brief 获取半边
   * @param index 半边索引
   * @return 半边引用
   */
  const HalfEdge &get_halfedge(size_t index) const;

  /**
   * @brief 获取所有半边
   * @return 半边数组
   */
  const std::vector<HalfEdge> &get_halfedges() const;

  /**
   * @brief 获取顶点到出边的映射
   * @return 顶点到出边的映射
   */
  const std::vector<std::vector<size_t>> &get_vertex_to_halfedges() const;

  /**
   * @brief 获取面片到半边的映射
   * @return 面片到半边的映射
   */
  const std::vector<size_t> &get_face_to_halfedge() const;

private:
  std::vector<HalfEdge> halfedges_;                      // 所有半边
  std::vector<std::vector<size_t>> vertex_to_halfedges_; // 顶点到出边的映射
  std::vector<size_t> face_to_halfedge_;                 // 面片到半边的映射

  /**
   * @brief 构建半边结构
   * @param faces 面片数组
   * @param num_vertices 顶点数量
   */
  void build(const std::vector<std::array<size_t, 3>> &faces,
             size_t num_vertices);
};

} // namespace mesh_graph_cut_cpp
