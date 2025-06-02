#include "sample.h"
#include <iostream>
#include <omp.h>

// 计算两点之间的欧氏距离的平方
inline float compute_distance_squared(const float *p1, const float *p2,
                                      int dim) {
  float dist = 0.0f;
#pragma omp simd reduction(+ : dist)
  for (int d = 0; d < dim; ++d) {
    float diff = p1[d] - p2[d];
    dist += diff * diff;
  }
  return dist;
}

std::vector<int> farthest_point_sampling(py::array_t<float> points,
                                         int sample_point_num) {
  auto points_buf = points.request();
  if (points_buf.ndim != 2) {
    throw std::runtime_error("Input points must be a 2D array");
  }

  const int num_points = points_buf.shape[0];
  const int dim = points_buf.shape[1];
  const float *points_ptr = static_cast<float *>(points_buf.ptr);

  if (sample_point_num > num_points) {
    throw std::runtime_error(
        "Sample size cannot be larger than the number of points");
  }

  // 初始化结果数组和距离数组
  std::vector<int> sampled_indices(sample_point_num);
  std::vector<float> distances(num_points,
                               std::numeric_limits<float>::infinity());

  // 使用随机数生成器选择第一个点
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, num_points - 1);
  sampled_indices[0] = dis(gen);

  std::cout << "[INFO][sample::farthest_point_sampling]" << std::endl;
  std::cout << "\t start sample fps points..." << std::endl;

  // 主循环
  for (int i = 1; i < sample_point_num; ++i) {
    const float *farthest = points_ptr + sampled_indices[i - 1] * dim;
    float max_dist = -std::numeric_limits<float>::infinity();
    int max_idx = 0;

// 并行计算所有点到当前最远点的距离
#pragma omp parallel
    {
      float local_max_dist = -std::numeric_limits<float>::infinity();
      int local_max_idx = 0;

#pragma omp for nowait
      for (int j = 0; j < num_points; ++j) {
        const float *current = points_ptr + j * dim;
        float dist = compute_distance_squared(current, farthest, dim);
        distances[j] = std::min(distances[j], dist);

        if (distances[j] > local_max_dist) {
          local_max_dist = distances[j];
          local_max_idx = j;
        }
      }

// 规约操作：找到全局最大距离
#pragma omp critical
      {
        if (local_max_dist > max_dist) {
          max_dist = local_max_dist;
          max_idx = local_max_idx;
        }
      }
    }

    sampled_indices[i] = max_idx;

    /*
    // 打印进度（每10%打印一次）
    if ((i + 1) % (sample_point_num / 10) == 0) {
      std::cout << "\t " << (i + 1) * 100 / sample_point_num << "% completed"
                << std::endl;
    }
    */
  }

  return sampled_indices;
}
