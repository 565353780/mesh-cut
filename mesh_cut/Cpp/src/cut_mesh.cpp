#include "cut_mesh.h"
#include <algorithm>
#include <map>
#include <mcut/mcut.h>
#include <mio/mio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS 1

#ifdef _WIN32
#pragma warning(disable : 26812) // Unscoped enums from mcut.h
#endif                           // _WIN32
#endif

#define my_assert(cond)                                                        \
  if (!(cond)) {                                                               \
    fprintf(stderr, "MCUT error: %s\n", #cond);                                \
    std::exit(1);                                                              \
  }

void MCAPI_PTR mcDebugOutput(McDebugSource source, McDebugType type,
                             McUint32 id, McDebugSeverity severity,
                             size_t length, const char *message,
                             const void *userParam);

void cutMesh(const std::string &mesh_file_path,
             const std::string &cut_mesh_file_path) {
  MioMesh srcMesh = {
      nullptr, // pVertices
      nullptr, // pNormals
      nullptr, // pTexCoords
      nullptr, // pFaceSizes
      nullptr, // pFaceVertexIndices
      nullptr, // pFaceVertexTexCoordIndices
      nullptr, // pFaceVertexNormalIndices
      0,       // numVertices
      0,       // numNormals
      0,       // numTexCoords
      0,       // numFaces
  };

  MioMesh cutMesh = srcMesh;

  mioReadOBJ(mesh_file_path.c_str(), &srcMesh.pVertices, &srcMesh.pNormals,
             &srcMesh.pTexCoords, &srcMesh.pFaceSizes,
             &srcMesh.pFaceVertexIndices, &srcMesh.pFaceVertexTexCoordIndices,
             &srcMesh.pFaceVertexNormalIndices, &srcMesh.numVertices,
             &srcMesh.numNormals, &srcMesh.numTexCoords, &srcMesh.numFaces);

  mioReadOBJ(cut_mesh_file_path.c_str(), &cutMesh.pVertices, &cutMesh.pNormals,
             &cutMesh.pTexCoords, &cutMesh.pFaceSizes,
             &cutMesh.pFaceVertexIndices, &cutMesh.pFaceVertexTexCoordIndices,
             &cutMesh.pFaceVertexNormalIndices, &cutMesh.numVertices,
             &cutMesh.numNormals, &cutMesh.numTexCoords, &cutMesh.numFaces);

  //
  // create a context
  //
  McContext context = MC_NULL_HANDLE;

  // a debug context is created in case you load your own (possibly faulty)
  // meshes
  McResult status = mcCreateContext(&context, MC_DEBUG);

  my_assert(status == MC_NO_ERROR);

  //
  // config debug output
  //

  McSize numBytes = 0;
  McFlags contextFlags;

  status = mcGetInfo(context, MC_CONTEXT_FLAGS, 0, nullptr, &numBytes);

  my_assert(status == MC_NO_ERROR);

  my_assert(sizeof(McFlags) == numBytes);

  status =
      mcGetInfo(context, MC_CONTEXT_FLAGS, numBytes, &contextFlags, nullptr);

  my_assert(status == MC_NO_ERROR);

  if (contextFlags & MC_DEBUG) { // did the user enable debugging mode?
    mcDebugMessageCallback(context, mcDebugOutput, nullptr);
    mcDebugMessageControl(context, McDebugSource::MC_DEBUG_SOURCE_ALL,
                          McDebugType::MC_DEBUG_TYPE_ALL,
                          McDebugSeverity::MC_DEBUG_SEVERITY_ALL, true);
  }

  //
  //  do the cutting (boolean ops)
  //

  printf("\nInputs: \n\tSolid-A = %s'.\n\tSolid-B = '%s'\n\n",
         mesh_file_path.c_str(), cut_mesh_file_path.c_str());

  // We can either let MCUT compute all possible meshes (including patches
  // etc.), or we can constrain the library runtime to compute exactly the
  // boolean op mesh we want. This 'constrained' case is done with the flags
  // that follow below.
  //

  printf("operation start\n");

  status = mcDispatch(
      context,
      MC_DISPATCH_VERTEX_ARRAY_DOUBLE | // vertices are in array of doubles
          MC_DISPATCH_ENFORCE_GENERAL_POSITION | // perturb if necessary
          MC_DISPATCH_FILTER_FRAGMENT_SEALING_NONE,
      // source mesh
      srcMesh.pVertices, srcMesh.pFaceVertexIndices, srcMesh.pFaceSizes,
      srcMesh.numVertices, srcMesh.numFaces,
      // cut mesh
      cutMesh.pVertices, cutMesh.pFaceVertexIndices, cutMesh.pFaceSizes,
      cutMesh.numVertices, cutMesh.numFaces);

  my_assert(status == MC_NO_ERROR);

  //
  // query the number of available fragments
  // NOTE: a boolean operation shall always give fragments as output
  //

  McUint32 connectedComponentCount = 0;
  status =
      mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_FRAGMENT, 0,
                               NULL, &connectedComponentCount);
  my_assert(status == MC_NO_ERROR);

  if (connectedComponentCount == 0) {
    fprintf(stdout, "no connected components found\n");
    exit(0);
  }

  std::vector<McConnectedComponent> connectedComponents(connectedComponentCount,
                                                        MC_NULL_HANDLE);

  status = mcGetConnectedComponents(
      context, MC_CONNECTED_COMPONENT_TYPE_FRAGMENT,
      (McUint32)connectedComponents.size(), connectedComponents.data(), NULL);

  my_assert(status == MC_NO_ERROR);

  //
  // query the data of the output connected component from MCUT
  //

  McConnectedComponent cc = connectedComponents[0];

  //
  // vertices
  //

  numBytes = 0;
  status = mcGetConnectedComponentData(
      context, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, 0, NULL,
      &numBytes);

  my_assert(status == MC_NO_ERROR);

  McUint32 ccVertexCount = (McUint32)(numBytes / (sizeof(McDouble) * 3));
  std::vector<McDouble> ccVertices((McSize)ccVertexCount * 3u, 0);
  status = mcGetConnectedComponentData(
      context, cc, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, numBytes,
      (void *)ccVertices.data(), NULL);

  my_assert(status == MC_NO_ERROR);

  //
  // faces
  //
  numBytes = 0;

  // triangulated faces
  status = mcGetConnectedComponentData(
      context, cc, MC_CONNECTED_COMPONENT_DATA_FACE_TRIANGULATION, 0, NULL,
      &numBytes);

  my_assert(status == MC_NO_ERROR);

  std::vector<McUint32> ccFaceIndices(numBytes / sizeof(McUint32), 0);
  status = mcGetConnectedComponentData(
      context, cc, MC_CONNECTED_COMPONENT_DATA_FACE_TRIANGULATION, numBytes,
      ccFaceIndices.data(), NULL);

  my_assert(status == MC_NO_ERROR);

  std::vector<McUint32> ccFaceSizes(ccFaceIndices.size() / 3, 3);
  /// ------------------------------------------------------------------------------------

  // Here we show, how to know when connected components pertain particular
  // boolean operations.
  McPatchLocation patchLocation = (McPatchLocation)0;

  status = mcGetConnectedComponentData(
      context, cc, MC_CONNECTED_COMPONENT_DATA_PATCH_LOCATION,
      sizeof(McPatchLocation), &patchLocation, NULL);
  my_assert(status == MC_NO_ERROR);

  McFragmentLocation fragmentLocation = (McFragmentLocation)0;
  status = mcGetConnectedComponentData(
      context, cc, MC_CONNECTED_COMPONENT_DATA_FRAGMENT_LOCATION,
      sizeof(McFragmentLocation), &fragmentLocation, NULL);
  my_assert(status == MC_NO_ERROR);

  //
  // reverse the vertex winding order, if required
  //
  if ((fragmentLocation == MC_FRAGMENT_LOCATION_BELOW) &&
      (patchLocation == MC_PATCH_LOCATION_OUTSIDE)) {
    std::reverse(ccFaceIndices.begin(), ccFaceIndices.end());
  }

  //
  // save connected component (mesh) to an .obj file
  //

  auto extract_fname = [](const std::string &full_path) {
    // get filename
    std::string base_filename =
        full_path.substr(full_path.find_last_of("/\\") + 1);
    // remove extension from filename
    std::string::size_type const p(base_filename.find_last_of('.'));
    std::string file_without_extension = base_filename.substr(0, p);
    return file_without_extension;
  };

  const std::string fpath("./output/" + extract_fname(mesh_file_path.c_str()) +
                          "_" + extract_fname(cut_mesh_file_path.c_str()) +
                          ".obj");

  mioWriteOBJ(fpath.c_str(), ccVertices.data(),
              nullptr, // pNormals
              nullptr, // pTexCoords
              ccFaceSizes.data(), ccFaceIndices.data(),
              nullptr, // pFaceVertexTexCoordIndices
              nullptr, // pFaceVertexNormalIndices
              ccVertexCount,
              0, // numNormals
              0, // numTexCoords
              (McUint32)ccFaceSizes.size());

  //
  // free connected component data
  //
  status = mcReleaseConnectedComponents(context,
                                        (McUint32)connectedComponents.size(),
                                        connectedComponents.data());

  my_assert(status == MC_NO_ERROR);

  //
  // We no longer need the mem of input meshes, so we can free it!
  //
  mioFreeMesh(&srcMesh);
  mioFreeMesh(&cutMesh);

  //
  // destroy context
  //
  status = mcReleaseContext(context);

  my_assert(status == MC_NO_ERROR);
}

void MCAPI_PTR mcDebugOutput(McDebugSource source, McDebugType type,
                             McUint32 id, McDebugSeverity severity,
                             size_t length, const char *message,
                             const void *userParam) {
  std::string debug_src;
  switch (source) {
  case MC_DEBUG_SOURCE_API:
    debug_src = "API";
    break;
  case MC_DEBUG_SOURCE_KERNEL:
    debug_src = "KERNEL";
    break;
  case MC_DEBUG_SOURCE_FRONTEND:
    debug_src = "FRONTEND";
  case MC_DEBUG_SOURCE_ALL:
  case MC_DEBUG_SOURCE_IGNORE:
    break;
  }

  std::string debug_type;
  switch (type) {
  case MC_DEBUG_TYPE_ERROR:
    debug_type = "ERROR";
    break;
  case MC_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
    debug_type = "DEPRECATION";
    break;
  case MC_DEBUG_TYPE_OTHER:
    // printf("Type: Other");
    debug_type = "OTHER";
    break;
  case MC_DEBUG_TYPE_ALL:
  case MC_DEBUG_TYPE_IGNORE:
    break;
  }

  std::string severity_str;

  switch (severity) {
  case MC_DEBUG_SEVERITY_HIGH:
    severity_str = "HIGH";
    break;
  case MC_DEBUG_SEVERITY_MEDIUM:
    severity_str = "MEDIUM";
    break;
  case MC_DEBUG_SEVERITY_LOW:
    severity_str = "LOW";
    break;
  case MC_DEBUG_SEVERITY_NOTIFICATION:
    severity_str = "NOTIFICATION";
    break;
  case MC_DEBUG_SEVERITY_ALL:
    break;
  }

  printf("MCUT[%d:%p,%s:%s:%s:%zu] %s\n", id, userParam, debug_src.c_str(),
         debug_type.c_str(), severity_str.c_str(), length, message);
}
