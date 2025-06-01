#pragma once

#include "nanoflann.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace mesh_graph_cut_cpp {

/**
 * @brief 使用nanoflann库实现的KD树
 */
class KDTree {
private:
  // 点云适配器，用于nanoflann库
  struct PointCloudAdapter {
    const std::vector<std::array<double, 3>> &points;

    PointCloudAdapter(const std::vector<std::array<double, 3>> &pts)
        : points(pts) {}

    // 必须的接口函数
    inline size_t kdtree_get_point_count() const { return points.size(); }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
      return points[idx][dim];
    }

    // 可选的边界盒计算函数
    template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
  };

  // 使用nanoflann的KD树类型
  using KDTreeType = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, PointCloudAdapter>,
      PointCloudAdapter,
      3,     // 维度
      size_t // 索引类型
      >;

  PointCloudAdapter points_adapter;
  std::unique_ptr<KDTreeType> index;
  const std::vector<std::array<double, 3>> &points_;

public:
  /**
   * @brief 构造函数
   * @param points 3D点集
   */
  KDTree(const std::vector<std::array<double, 3>> &points)
      : points_adapter(points), points_(points) {
    // 创建KD树索引
    index = std::make_unique<KDTreeType>(
        3, points_adapter, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index->buildIndex();
  }

  /**
   * @brief 计算两点之间的欧氏距离
   * @param a 第一个点
   * @param b 第二个点
   * @return 欧氏距离
   */
  static double distance(const std::array<double, 3> &a,
                         const std::array<double, 3> &b) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  /**
   * @brief 查找给定点的最近邻
   * @param query 查询点
   * @return 最近邻的索引
   */
  size_t nearest(const std::array<double, 3> &query) const {
    size_t index_result;
    double distance_result;

    nanoflann::KNNResultSet<double> result_set(1);
    result_set.init(&index_result, &distance_result);

    index->findNeighbors(result_set, query.data(),
                         nanoflann::SearchParameters());

    return index_result;
  }

  /**
   * @brief 查找给定半径内的所有点
   * @param query 查询点
   * @param radius 搜索半径
   * @return 半径内点的索引列表
   */
  std::vector<size_t> radius_search(const std::array<double, 3>& query, 
                                    double radius) const {
    std::vector<nanoflann::ResultItem<size_t, double>> matches;

    // 注意：nanoflann需要平方半径
    const double search_radius_sq = radius * radius;

    // 执行半径搜索
    const size_t num_matches = index->radiusSearch(
        query.data(), search_radius_sq, matches, nanoflann::SearchParameters());

    // 提取索引
    std::vector<size_t> result;
    result.reserve(num_matches);
    for (size_t i = 0; i < num_matches; ++i) {
      result.push_back(matches[i].first);
    }

    return result;
  }
};

} // namespace mesh_graph_cut_cpp
