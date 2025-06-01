#include "halfedge.h"
#include <functional>
#include <iostream>
#include <unordered_map>

namespace mesh_graph_cut_cpp {

HalfEdgeStructure::HalfEdgeStructure(
    const std::vector<std::array<size_t, 3>> &faces, size_t num_vertices) {
  build(faces, num_vertices);
}

void HalfEdgeStructure::build(const std::vector<std::array<size_t, 3>> &faces,
                              size_t num_vertices) {
  // 初始化数据结构
  vertex_to_halfedges_.resize(num_vertices);
  face_to_halfedge_.resize(faces.size(), static_cast<size_t>(-1));

  // 用于查找对偶半边的映射
  // 为std::pair<size_t, size_t>定义哈希函数
  struct PairHash {
    size_t operator()(const std::pair<size_t, size_t> &p) const {
      // 使用Boost的哈希组合技术
      size_t h1 = std::hash<size_t>{}(p.first);
      size_t h2 = std::hash<size_t>{}(p.second);
      return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
  };

  // 使用自定义哈希函数的unordered_map
  std::unordered_map<std::pair<size_t, size_t>, size_t, PairHash> edge_map;

  // 为每个面片创建半边
  for (size_t face_idx = 0; face_idx < faces.size(); ++face_idx) {
    const auto &face = faces[face_idx];
    size_t base_idx = halfedges_.size();

    // 创建面片的三条半边
    for (size_t i = 0; i < 3; ++i) {
      size_t next_i = (i + 1) % 3;
      size_t origin = face[i];
      size_t destination = face[next_i];

      // 创建半边
      halfedges_.emplace_back(origin, destination, face_idx,
                              base_idx + (i + 1) % 3, static_cast<size_t>(-1));

      // 更新顶点到半边的映射
      vertex_to_halfedges_[origin].push_back(base_idx + i);

      // 记录半边用于后续查找对偶边
      edge_map[{origin, destination}] = base_idx + i;
    }

    // 记录面片的第一条半边
    face_to_halfedge_[face_idx] = base_idx;
  }

  // 设置对偶半边
  for (size_t i = 0; i < halfedges_.size(); ++i) {
    HalfEdge &he = halfedges_[i];
    auto twin_it = edge_map.find({he.destination, he.origin});

    if (twin_it != edge_map.end()) {
      he.twin = twin_it->second;
      halfedges_[twin_it->second].twin = i;
    }
  }
}

std::vector<size_t>
HalfEdgeStructure::get_vertex_neighbors(size_t vertex_index) const {
  std::vector<size_t> neighbors;
  std::unordered_set<size_t> visited;

  // 遍历从顶点出发的所有半边
  for (size_t he_idx : vertex_to_halfedges_[vertex_index]) {
    const HalfEdge &he = halfedges_[he_idx];
    if (visited.find(he.destination) == visited.end()) {
      neighbors.push_back(he.destination);
      visited.insert(he.destination);
    }
  }

  return neighbors;
}

std::vector<size_t>
HalfEdgeStructure::get_vertex_faces(size_t vertex_index) const {
  std::vector<size_t> faces;
  std::unordered_set<size_t> visited;

  // 遍历从顶点出发的所有半边
  for (size_t he_idx : vertex_to_halfedges_[vertex_index]) {
    const HalfEdge &he = halfedges_[he_idx];
    if (visited.find(he.face) == visited.end()) {
      faces.push_back(he.face);
      visited.insert(he.face);
    }
  }

  return faces;
}

std::vector<size_t>
HalfEdgeStructure::get_face_neighbors(size_t face_index) const {
  std::vector<size_t> neighbors;
  std::array<size_t, 3> face_halfedges = get_face_halfedges(face_index);

  // 遍历面片的三条半边
  for (size_t he_idx : face_halfedges) {
    const HalfEdge &he = halfedges_[he_idx];

    // 如果有对偶半边，则对应面片是邻居
    if (he.twin != static_cast<size_t>(-1)) {
      const HalfEdge &twin = halfedges_[he.twin];
      neighbors.push_back(twin.face);
    }
  }

  return neighbors;
}

std::vector<size_t>
HalfEdgeStructure::get_outgoing_halfedges(size_t vertex_index) const {
  return vertex_to_halfedges_[vertex_index];
}

std::array<size_t, 3>
HalfEdgeStructure::get_face_halfedges(size_t face_index) const {
  size_t first = face_to_halfedge_[face_index];
  size_t second = halfedges_[first].next;
  size_t third = halfedges_[second].next;
  return {first, second, third};
}

const HalfEdge &HalfEdgeStructure::get_halfedge(size_t index) const {
  return halfedges_[index];
}

const std::vector<HalfEdge> &HalfEdgeStructure::get_halfedges() const {
  return halfedges_;
}

const std::vector<std::vector<size_t>> &
HalfEdgeStructure::get_vertex_to_halfedges() const {
  return vertex_to_halfedges_;
}

const std::vector<size_t> &HalfEdgeStructure::get_face_to_halfedge() const {
  return face_to_halfedge_;
}

} // namespace mesh_graph_cut_cpp
