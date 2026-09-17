"""
view_pointcloud.py
--------------------
PHASE 5 module for ZenX Drone3D (SIH26158).

What this does:
    Loads the sparse 3D point cloud that COLMAP produced in Phase 4,
    removes obvious noisy/outlier points, and opens an interactive
    window where you can rotate and zoom around the real reconstructed
    scene.

Why matplotlib instead of Open3D:
    Open3D's official pip package does not yet publish wheels for very
    new Python versions (e.g. Python 3.13 at the time of writing), so
    `pip install open3d` can fail with "no matching distribution"
    depending on your installed Python version. matplotlib is far more
    universally compatible and is usually already installed. It is
    less visually polished than Open3D but gives a real, working,
    honest 3D view -- which is what matters for the prototype.

Why TXT format from COLMAP (not PLY/binary):
    We export COLMAP's model as plain text (--output_type TXT) instead
    of PLY or binary, so this script can parse it with pure Python --
    no extra point-cloud library needed, avoiding another possible
    version-compatibility dead end.

Why remove outliers:
    Uses the same idea as standard "statistical outlier removal": for
    each point, find its nearest neighbors (via scipy's fast KD-tree)
    and remove points whose average neighbor distance is much larger
    than typical -- i.e. isolated points not part of the real surface.

Status: IMPLEMENTED (COLMAP TXT export, outlier removal via scipy,
interactive 3D view + screenshot saving via matplotlib).
Not yet implemented: mesh generation (needs a denser point cloud from
GPU-based dense reconstruction, which this machine's hardware doesn't
support -- documented as future work).
"""

import argparse
import os
import sys
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from scipy.spatial import cKDTree


def convert_colmap_model_to_txt(colmap_exe: str, model_dir: str, txt_out_dir: str):
    """
    Converts COLMAP's binary sparse model into plain-text files
    (cameras.txt, images.txt, points3D.txt) that we can parse with
    pure Python -- no extra library required.
    """
    os.makedirs(txt_out_dir, exist_ok=True)
    print(f"Converting COLMAP model at {model_dir} to TXT format...")
    result = subprocess.run([
        colmap_exe, "model_converter",
        "--input_path", model_dir,
        "--output_path", txt_out_dir,
        "--output_type", "TXT",
    ])
    if result.returncode != 0:
        print("[ERROR] model_converter failed. Check that --model-dir points "
              "to a valid COLMAP sparse model folder (containing cameras.bin, "
              "images.bin, points3D.bin).")
        sys.exit(1)
    print(f"[OK] Wrote text model to: {txt_out_dir}")


def load_points3d_txt(points3d_txt_path: str):
    """
    Parses COLMAP's points3D.txt format:
        POINT3D_ID X Y Z R G B ERROR TRACK[](IMAGE_ID, POINT2D_IDX)...
    Lines starting with '#' are comments/headers and are skipped.
    Returns (xyz, rgb) as numpy arrays.
    """
    if not os.path.exists(points3d_txt_path):
        print(f"[ERROR] File not found: {points3d_txt_path}")
        sys.exit(1)

    xyz = []
    rgb = []
    with open(points3d_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            xyz.append([x, y, z])
            rgb.append([r, g, b])

    xyz = np.array(xyz)
    rgb = np.array(rgb) / 255.0  # normalize to 0-1 for matplotlib

    print(f"Loaded point cloud: {len(xyz)} points")
    if len(xyz) == 0:
        print("[ERROR] Point cloud is empty. Nothing to visualize.")
        sys.exit(1)

    return xyz, rgb


def remove_statistical_outliers(xyz: np.ndarray, rgb: np.ndarray,
                                  k_neighbors: int, std_ratio: float):
    """
    For each point, computes the average distance to its k nearest
    neighbors. Points whose average neighbor distance is more than
    `std_ratio` standard deviations above the overall mean are removed
    as outliers (isolated points, not part of the real surface).
    """
    n_points = len(xyz)
    k = min(k_neighbors, n_points - 1)
    if k < 1:
        print("[WARNING] Too few points for outlier removal, skipping.")
        return xyz, rgb

    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=k + 1)  # includes the point itself (distance 0)
    mean_neighbor_dist = distances[:, 1:].mean(axis=1)  # exclude self

    overall_mean = mean_neighbor_dist.mean()
    overall_std = mean_neighbor_dist.std()
    threshold = overall_mean + std_ratio * overall_std

    keep_mask = mean_neighbor_dist <= threshold
    removed_count = n_points - keep_mask.sum()

    print(f"Removed {removed_count} outlier points "
          f"({removed_count / n_points * 100:.1f}% of total)")
    print(f"Remaining: {keep_mask.sum()} points")

    return xyz[keep_mask], rgb[keep_mask]


def plot_point_cloud(xyz: np.ndarray, rgb: np.ndarray, screenshot_path: str = None,
                      show_window: bool = True):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=2)
    ax.set_title("ZenX Drone3D - Sparse Point Cloud (Phase 5)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    max_range = np.array([
        xyz[:, 0].max() - xyz[:, 0].min(),
        xyz[:, 1].max() - xyz[:, 1].min(),
        xyz[:, 2].max() - xyz[:, 2].min(),
    ]).max() / 2.0
    mid_x = (xyz[:, 0].max() + xyz[:, 0].min()) / 2.0
    mid_y = (xyz[:, 1].max() + xyz[:, 1].min()) / 2.0
    mid_z = (xyz[:, 2].max() + xyz[:, 2].min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if screenshot_path:
        fig.savefig(screenshot_path, dpi=150)
        print(f"Saved preview image to: {screenshot_path}")

    if show_window:
        print("\nOpening interactive 3D viewer...")
        print("Controls: left-click+drag to rotate, scroll to zoom "
              "(or use the toolbar zoom tool). Close the window to finish.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert, clean, and view a COLMAP sparse point cloud with matplotlib."
    )
    parser.add_argument("--model-dir", required=True,
                         help="Path to COLMAP sparse model folder, e.g. colmap\\sparse\\0")
    parser.add_argument("--colmap-exe", default="colmap",
                         help="Path to COLMAP executable (e.g. C:\\Tools\\COLMAP\\COLMAP.bat)")
    parser.add_argument("--txt-out-dir", default=None,
                         help="Where to save the converted TXT files "
                              "(default: <model-dir>\\txt_export)")
    parser.add_argument("--k-neighbors", type=int, default=15,
                         help="Neighbors to check per point for outlier removal (default: 15)")
    parser.add_argument("--std-ratio", type=float, default=2.0,
                         help="Outlier distance threshold in std-deviations (default: 2.0)")
    parser.add_argument("--no-window", action="store_true",
                         help="Skip opening the interactive window (just convert, clean, "
                              "and save a screenshot -- useful for quick verification)")
    parser.add_argument("--screenshot", default=None,
                         help="Optional path to save a static preview PNG")

    args = parser.parse_args()

    txt_out_dir = args.txt_out_dir or os.path.join(args.model_dir, "txt_export")

    convert_colmap_model_to_txt(args.colmap_exe, args.model_dir, txt_out_dir)
    xyz, rgb = load_points3d_txt(os.path.join(txt_out_dir, "points3D.txt"))
    xyz_clean, rgb_clean = remove_statistical_outliers(
        xyz, rgb, args.k_neighbors, args.std_ratio
    )

    plot_point_cloud(
        xyz_clean, rgb_clean,
        screenshot_path=args.screenshot,
        show_window=not args.no_window,
    )