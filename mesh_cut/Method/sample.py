import trimesh
import numpy as np

from tqdm import trange
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point


def toFPSIdxs(points: np.ndarray, sample_point_num: int) -> np.ndarray:
    N = points.shape[0]
    sampled_indices = np.zeros(sample_point_num, dtype=int)
    distances = np.full(N, np.inf)

    # 初始化：从一个随机点开始
    sampled_indices[0] = np.random.randint(N)
    farthest = points[sampled_indices[0]]

    print("[INFO][sample::toFPSIdxs]")
    print("\t start sample fps points...")
    for i in trange(1, sample_point_num):
        dist = np.linalg.norm(points - farthest, axis=1)
        distances = np.minimum(distances, dist)
        sampled_indices[i] = np.argmax(distances)
        farthest = points[sampled_indices[i]]

    return sampled_indices

def sampleBoundaryMesh(
    loop_coords: np.ndarray,
    interior_spacing: float = 0.01,
) -> trimesh.Trimesh:
    centroid = loop_coords.mean(axis=0)
    centered = loop_coords - centroid

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[2]

    e_x = Vt[0]
    e_y = Vt[1]
    R = np.stack([e_x, e_y, normal], axis=0)

    pts_2d_raw = (R @ centered.T).T[:, :2]

    bb_min = pts_2d_raw.min(axis=0)
    bb_max = pts_2d_raw.max(axis=0)
    bb_center = (bb_min + bb_max) / 2.0
    bb_extent = bb_max - bb_min
    scale = bb_extent.max()

    pts_2d = (pts_2d_raw - bb_center) / scale

    polygon = Polygon(pts_2d)

    xmin, ymin, xmax, ymax = polygon.bounds
    xs = np.arange(xmin + interior_spacing, xmax, interior_spacing)
    ys = np.arange(ymin + interior_spacing, ymax, interior_spacing)
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    interior_mask = np.array([polygon.contains(Point(p)) for p in grid_pts])
    interior_pts_2d = grid_pts[interior_mask]

    all_pts_2d = np.vstack([pts_2d, interior_pts_2d])

    tri = Delaunay(all_pts_2d)
    faces = tri.simplices

    valid_mask = np.array([
        polygon.contains(Point(all_pts_2d[f].mean(axis=0))) for f in faces
    ])
    faces = faces[valid_mask]

    all_pts_2d_restored = all_pts_2d * scale + bb_center
    all_pts_3d_local = np.column_stack([
        all_pts_2d_restored, np.zeros(len(all_pts_2d_restored))
    ])
    vertices = (R.T @ all_pts_3d_local.T).T + centroid

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
