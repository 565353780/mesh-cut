#include "cut_mesh.h"
#include "mcut/mcut.h"

void cutMesh(const torch::Tensor &vertices, const torch::Tensor &triangles,
             const torch::Tensor &cut_vertices,
             const torch::Tensor &cut_triangles) {
  McFloat cubeVertices[] = {
      -5, -5, 5,  // 0
      5,  -5, 5,  // 1
      5,  5,  5,  // 2
      -5, 5,  5,  // 3
      -5, -5, -5, // 4
      5,  -5, -5, // 5
      5,  5,  -5, // 6
      -5, 5,  -5  // 7
  };
  McUint32 cubeFaces[] = {
      0, 1, 2, 3, // 0
      7, 6, 5, 4, // 1
      1, 5, 6, 2, // 2
      0, 3, 7, 4, // 3
      3, 2, 6, 7, // 4
      4, 5, 1, 0  // 5
  };
  McUint32 cubeFaceSizes[] = {4, 4, 4, 4, 4, 4};
  McUint32 numCubeVertices = 8;
  McUint32 numCubeFaces = 6;

  return;
}
