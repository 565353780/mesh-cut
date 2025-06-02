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

// 计算三角形面积
inline float compute_triangle_area(const float *v0, const float *v1,
                                   const float *v2, int dim) {
  if (dim != 3) {
    throw std::runtime_error(
        "Triangle area calculation only supports 3D points");
  }

  // 计算两条边的向量
  float edge1[3], edge2[3];
  for (int d = 0; d < dim; ++d) {
    edge1[d] = v1[d] - v0[d];
    edge2[d] = v2[d] - v0[d];
  }

  // 计算叉积
  float cross[3];
  cross[0] = edge1[1] * edge2[2] - edge1[2] * edge2[1];
  cross[1] = edge1[2] * edge2[0] - edge1[0] * edge2[2];
  cross[2] = edge1[0] * edge2[1] - edge1[1] * edge2[0];

  // 计算叉积的长度（面积的两倍）
  float area = 0.0f;
  for (int d = 0; d < 3; ++d) {
    area += cross[d] * cross[d];
  }

  return 0.5f * std::sqrt(area);
}

// 在三角形内部均匀采样点
inline void sample_triangle_uniform(const float *v0, const float *v1,
                                    const float *v2, float *point, int dim,
                                    std::mt19937 &gen) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  // 生成重心坐标
  float r1 = dist(gen);
  float r2 = dist(gen);

  // 确保在三角形内部
  if (r1 + r2 > 1.0f) {
    r1 = 1.0f - r1;
    r2 = 1.0f - r2;
  }

  float r0 = 1.0f - r1 - r2;

  // 计算采样点坐标
  for (int d = 0; d < dim; ++d) {
    point[d] = r0 * v0[d] + r1 * v1[d] + r2 * v2[d];
  }
}

py::array_t<float> toSubMeshSamplePoints(py::array_t<float> vertices,
                                         py::array_t<int> triangles,
                                         const std::vector<std::vector<size_t>>& face_groups) {
  // 获取输入数组的信息
  auto vertices_buf = vertices.request();
  auto triangles_buf = triangles.request();

  if (vertices_buf.ndim != 2 || triangles_buf.ndim != 2) {
    throw std::runtime_error("Input arrays have incorrect dimensions");
  }

  const int num_vertices = vertices_buf.shape[0];
  const int dim = vertices_buf.shape[1];
  const float *vertices_ptr = static_cast<float *>(vertices_buf.ptr);
  const int *triangles_ptr = static_cast<int *>(triangles_buf.ptr);

  const int NUM_SUBMESHES = face_groups.size();
  const int POINTS_PER_SUBMESH = 8192;

  // 创建结果数组
  std::vector<size_t> result_shape = {NUM_SUBMESHES, POINTS_PER_SUBMESH, dim};
  py::array_t<float> result(result_shape);
  auto result_buf = result.request();
  float *result_ptr = static_cast<float *>(result_buf.ptr);

  std::cout << "[INFO][sample::toSubMeshSamplePoints]" << std::endl;
  std::cout << "\t start uniform sampling on submeshes..." << std::endl;

  // 为每个子网格收集三角形和计算面积
  std::vector<std::vector<int>> submesh_triangles(NUM_SUBMESHES);
  std::vector<std::vector<float>> submesh_areas(NUM_SUBMESHES);
  std::vector<float> submesh_total_area(NUM_SUBMESHES, 0.0f);

  // 直接使用提供的面片组
  for (int submesh_id = 0; submesh_id < NUM_SUBMESHES; ++submesh_id) {
    const auto& face_indices = face_groups[submesh_id];
    
    // 将面片索引添加到子网格三角形列表
    for (size_t i = 0; i < face_indices.size(); ++i) {
      size_t f = face_indices[i];
      submesh_triangles[submesh_id].push_back(static_cast<int>(f));
      
      // 获取三角形顶点
      int v0_idx = triangles_ptr[f * 3];
      int v1_idx = triangles_ptr[f * 3 + 1];
      int v2_idx = triangles_ptr[f * 3 + 2];

      const float *v0 = vertices_ptr + v0_idx * dim;
      const float *v1 = vertices_ptr + v1_idx * dim;
      const float *v2 = vertices_ptr + v2_idx * dim;

      // 计算三角形面积
      float area = compute_triangle_area(v0, v1, v2, dim);
      submesh_areas[submesh_id].push_back(area);
      submesh_total_area[submesh_id] += area;
    }
  }

// 并行处理每个子网格
#pragma omp parallel
  {
    // 每个线程使用不同的随机数生成器种子
    std::random_device rd;
    std::mt19937 gen(rd());

#pragma omp for schedule(dynamic)
    for (int submesh_id = 0; submesh_id < NUM_SUBMESHES; ++submesh_id) {
      const auto &triangle_indices = submesh_triangles[submesh_id];
      const auto &areas = submesh_areas[submesh_id];
      const float total_area = submesh_total_area[submesh_id];

      if (triangle_indices.empty() || total_area <= 0.0f) {
        // 如果子网格为空或面积为零，填充零
        for (int i = 0; i < POINTS_PER_SUBMESH; ++i) {
          for (int d = 0; d < dim; ++d) {
            result_ptr[submesh_id * POINTS_PER_SUBMESH * dim + i * dim + d] =
                0.0f;
          }
        }
        continue;
      }

      // 计算每个三角形应分配的点数
      std::vector<int> points_per_triangle(triangle_indices.size());
      int total_assigned = 0;

      for (size_t i = 0; i < triangle_indices.size(); ++i) {
        // 根据面积比例分配点数
        float area_ratio = areas[i] / total_area;
        int num_points = static_cast<int>(area_ratio * POINTS_PER_SUBMESH);
        points_per_triangle[i] = num_points;
        total_assigned += num_points;
      }

      // 处理由于舍入导致的点数差异
      int remaining = POINTS_PER_SUBMESH - total_assigned;
      if (remaining > 0) {
        // 将剩余点分配给最大的三角形
        int max_area_idx = 0;
        float max_area = 0.0f;
        for (size_t i = 0; i < areas.size(); ++i) {
          if (areas[i] > max_area) {
            max_area = areas[i];
            max_area_idx = i;
          }
        }
        points_per_triangle[max_area_idx] += remaining;
      }

      // 在每个三角形内均匀采样
      int point_idx = 0;
      for (size_t i = 0; i < triangle_indices.size(); ++i) {
        int face_idx = triangle_indices[i];
        int num_points = points_per_triangle[i];

        // 获取三角形顶点
        int v0_idx = triangles_ptr[face_idx * 3];
        int v1_idx = triangles_ptr[face_idx * 3 + 1];
        int v2_idx = triangles_ptr[face_idx * 3 + 2];

        const float *v0 = vertices_ptr + v0_idx * dim;
        const float *v1 = vertices_ptr + v1_idx * dim;
        const float *v2 = vertices_ptr + v2_idx * dim;

        // 在三角形内均匀采样点
        for (int j = 0; j < num_points; ++j) {
          float *sample_point = result_ptr +
                                submesh_id * POINTS_PER_SUBMESH * dim +
                                point_idx * dim;
          sample_triangle_uniform(v0, v1, v2, sample_point, dim, gen);
          point_idx++;
        }
      }

      // 确保我们有足够的点
      if (point_idx < POINTS_PER_SUBMESH) {
        // 如果由于舍入误差导致点数不足，复制已有点
        for (int i = point_idx; i < POINTS_PER_SUBMESH; ++i) {
          int src_idx = i % point_idx;
          for (int d = 0; d < dim; ++d) {
            result_ptr[submesh_id * POINTS_PER_SUBMESH * dim + i * dim + d] =
                result_ptr[submesh_id * POINTS_PER_SUBMESH * dim +
                           src_idx * dim + d];
          }
        }
      }
    }
  }

  std::cout << "\t uniform sampling completed" << std::endl;
  return result;
}