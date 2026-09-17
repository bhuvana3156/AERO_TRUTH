"""
generate_solid_mesh.py
------------------------
PHASE 10 module for ZenX Drone3D (SIH26158).

What this does:
    Converts the sparse point cloud into an actual SOLID, ROTATABLE
    surface (a mesh with real triangular faces) instead of scattered
    dots -- the "3D model" you'd want to show judges. Fully automated,
    no manual clicking in external software required.

Why a different method than standard mesh tools:
    Standard tools (e.g. MeshLab's Poisson reconstruction) assume the
    point cloud roughly encloses a solid volume from many angles. A
    single-pass video's point cloud does NOT do that -- it only covers
    the surfaces visible along one camera path, so Poisson forces a
    fake closed blob around it (we tested this and got a bad thin
    wedge shape, not a usable model).

    Instead, this uses a technique suited to exactly this kind of data:
    1. PCA (Principal Component Analysis) finds the dominant flat-ish
       orientation of the scene (the two directions with the most
       spread), since a single-pass sweep is naturally more "spread
       out" than "enclosed."
    2. Points are projected onto that dominant plane (like flattening
       the scene into a 2D top-down/side-down view), then triangulated
       in 2D using Delaunay triangulation -- a standard, well-understood
       algorithm for connecting a set of points into non-overlapping
       triangles.
    3. Those 2D triangles are mapped back to the original 3D points,
       producing real connected triangular faces -- a genuine solid
       surface, not dots.
    4. GAP FILTERING: Delaunay triangulation can create long, fake
       "bridge" triangles across genuinely empty areas (where the
       camera never saw anything). We remove any triangle with an edge
       much longer than typical, so the model shows honest holes where
       data is missing, rather than fabricated surface.

Why this is more honest than forcing a closed shape:
    This will NOT look like a "globe you can spin and see every side
    of" -- because your single-pass video genuinely did not capture
    every side. What it WILL give you is a real, solid-looking,
    rotatable surface for everything that WAS captured, which is an
    accurate representation of your actual reconstruction -- exactly
    what a single-pass video can honestly produce.

Output:
    A single .ply file with real vertices AND faces (not just points),
    colored by the same confidence classification as Phase 9
    (green/yellow/red), saved to output/.

Status: IMPLEMENTED (PCA + Delaunay triangulation + gap filtering,
pure numpy/scipy -- no extra library, so no version-compatibility
risk). This is a simple, explainable heuristic suited to elongated
single-pass scans; it is not equivalent to full multi-view dense
meshing (which needs a CUDA GPU this machine does not have).
"""

import argparse
import os
import sys
import subprocess

import numpy as np
from scipy.spatial import Delaunay


def convert_colmap_model_to_txt(colmap_exe: str, model_dir: str, txt_out_dir: str):
    os.makedirs(txt_out_dir, exist_ok=True)
    print(f"Converting COLMAP model at {model_dir} to TXT format...")
    result = subprocess.run([
        colmap_exe, "model_converter",
        "--input_path", model_dir,
        "--output_path", txt_out_dir,
        "--output_type", "TXT",
    ])
    if result.returncode != 0:
        print("[ERROR] model_converter failed.")
        sys.exit(1)
    print(f"[OK] Wrote text model to: {txt_out_dir}")


def load_points_with_tracks(points3d_txt_path: str):
    if not os.path.exists(points3d_txt_path):
        print(f"[ERROR] File not found: {points3d_txt_path}")
        sys.exit(1)

    records = []
    with open(points3d_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            error = float(parts[7])
            track_length = (len(parts) - 8) // 2
            records.append({
                "x": x, "y": y, "z": z,
                "error": error, "track_length": track_length,
            })

    print(f"Loaded {len(records)} points")
    if len(records) < 4:
        print("[ERROR] Need at least 4 points to build a surface.")
        sys.exit(1)
    return records


def compute_confidence(records, max_track_for_full_score: int = 8,
                        max_error_for_zero_score: float = 2.0):
    for r in records:
        track_component = min(
            70.0, max(0.0, (r["track_length"] - 2) / (max_track_for_full_score - 2)) * 70.0
        )
        error_component = max(
            0.0, 30.0 - (min(r["error"], max_error_for_zero_score) / max_error_for_zero_score) * 30.0
        )
        score = round(min(100.0, track_component + error_component), 1)
        r["confidence_score"] = score
        if score >= 70:
            r["classification"] = "GREEN"
        elif score >= 40:
            r["classification"] = "YELLOW"
        else:
            r["classification"] = "RED"
    return records


def build_surface_mesh(records, gap_factor: float = 4.0):
    """
    Builds a triangulated surface from the point cloud via PCA + Delaunay,
    then removes triangles that bridge real gaps in the data (edges much
    longer than typical are considered "fake" connective triangles, not
    real surface, and are removed).
    """
    points = np.array([[r["x"], r["y"], r["z"]] for r in records])

    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    uv = centered @ eigvecs[:, :2]  # project onto the 2 dominant directions

    print("Running Delaunay triangulation...")
    tri = Delaunay(uv)
    print(f"Initial triangulation: {len(tri.simplices)} triangles")

    # Compute edge lengths (in real 3D space) for every triangle
    def triangle_max_edge(simplex):
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        return max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a))

    max_edges = np.array([triangle_max_edge(s) for s in tri.simplices])
    median_edge = np.median(max_edges)
    threshold = median_edge * gap_factor

    keep_mask = max_edges <= threshold
    kept_faces = tri.simplices[keep_mask]

    removed = len(tri.simplices) - len(kept_faces)
    print(f"Median triangle edge length: {median_edge:.4f}")
    print(f"Removed {removed} triangles bridging real gaps "
          f"(edge > {threshold:.4f}, {gap_factor}x median)")
    print(f"Final mesh: {len(kept_faces)} triangular faces, {len(points)} vertices")

    return points, kept_faces


def write_mesh_ply(points, faces, records, output_path: str):
    color_map = {
        "GREEN": (30, 180, 30),
        "YELLOW": (230, 200, 0),
        "RED": (215, 30, 30),
    }
    colors = np.array([color_map[r["classification"]] for r in records])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment ZenX Drone3D (SIH26158) - solid mesh, colored by confidence\n")
        f.write("comment green=high confidence, yellow=medium, red=uncertain\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"[OK] Wrote solid mesh to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a solid, rotatable mesh (real faces, not just points) from a COLMAP sparse model."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--colmap-exe", default="colmap")
    parser.add_argument("--txt-out-dir", default=None)
    parser.add_argument("--output", default="output\\zenx_drone3d_solid_model.ply")
    parser.add_argument("--gap-factor", type=float, default=4.0,
                         help="Triangles with an edge more than this many times the "
                              "median edge length are removed as fake gap-bridges (default: 4.0)")

    args = parser.parse_args()
    txt_out_dir = args.txt_out_dir or os.path.join(args.model_dir, "txt_export")

    convert_colmap_model_to_txt(args.colmap_exe, args.model_dir, txt_out_dir)
    records = load_points_with_tracks(os.path.join(txt_out_dir, "points3D.txt"))
    records = compute_confidence(records)
    points, faces = build_surface_mesh(records, args.gap_factor)
    write_mesh_ply(points, faces, records, args.output)

    print(f"\nOpen {args.output} in MeshLab, an online PLY viewer, or Windows 3D "
          f"software that supports PLY -- this is a real solid surface you can "
          f"rotate, not a point cloud.")