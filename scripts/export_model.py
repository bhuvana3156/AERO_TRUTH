"""
export_model.py
------------------
PHASE 10 module for ZenX Drone3D (SIH26158).

What this does:
    Produces the actual DELIVERABLE 3D model file: a standard .ply
    point cloud, colored by confidence classification (green/yellow/
    red from Phase 9), saved into output/. This is the file you hand
    to judges, attach to your report, or open in any standard 3D
    viewer -- not just something our own Python scripts can read.

Why this was missing before:
    Phases 1-9 processed data through COLMAP's own binary format and
    our scripts read it directly. That's fine for our pipeline, but it
    is NOT something a judge could open on their own laptop. A .ply
    file is the standard, portable format for exactly this situation --
    it's what real photogrammetry tools (COLMAP, Meshroom, Agisoft)
    all export, and it's openable in free tools like MeshLab, or even
    online PLY viewers with no install at all.

What's IN the exported file (real, measured data -- nothing invented):
    - Every point's real reconstructed X, Y, Z position (Phase 4)
    - A color for each point based on its REAL confidence classification
      (Phase 9): green = high confidence, yellow = medium, red = uncertain

Status: IMPLEMENTED (plain-text/ASCII PLY export, pure Python -- no
extra library needed, so no version-compatibility risk).
"""

import argparse
import os
import sys
import subprocess


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
    if not records:
        print("[ERROR] No points found.")
        sys.exit(1)
    return records


def compute_confidence(records, max_track_for_full_score: int = 8,
                        max_error_for_zero_score: float = 2.0):
    """Same formula as confidence_map.py (Phase 9) -- kept consistent."""
    for r in records:
        track_component = min(
            70.0,
            max(0.0, (r["track_length"] - 2) / (max_track_for_full_score - 2)) * 70.0
        )
        error_component = max(
            0.0,
            30.0 - (min(r["error"], max_error_for_zero_score) / max_error_for_zero_score) * 30.0
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


def write_confidence_ply(records, output_path: str):
    """
    Writes a standard ASCII .ply file with per-point color set to the
    confidence classification color. ASCII PLY is plain text -- no
    library needed, and it's readable/debuggable by just opening it
    in a text editor if something looks wrong.
    """
    color_map = {
        "GREEN": (30, 180, 30),
        "YELLOW": (230, 200, 0),
        "RED": (215, 30, 30),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment ZenX Drone3D (SIH26158) - sparse point cloud colored by confidence\n")
        f.write("comment green=high confidence, yellow=medium, red=uncertain\n")
        f.write(f"element vertex {len(records)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for r in records:
            cr, cg, cb = color_map[r["classification"]]
            f.write(f"{r['x']} {r['y']} {r['z']} {cr} {cg} {cb}\n")

    print(f"[OK] Wrote confidence-colored 3D model to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a standalone, confidence-colored .ply 3D model file."
    )
    parser.add_argument("--model-dir", required=True,
                         help="Path to COLMAP sparse model folder, e.g. colmap\\sparse\\0")
    parser.add_argument("--colmap-exe", default="colmap",
                         help="Path to COLMAP executable")
    parser.add_argument("--txt-out-dir", default=None,
                         help="Where to save converted TXT files (default: <model-dir>\\txt_export)")
    parser.add_argument("--output", default="output\\zenx_drone3d_model.ply",
                         help="Where to save the final .ply file (default: output\\zenx_drone3d_model.ply)")

    args = parser.parse_args()

    txt_out_dir = args.txt_out_dir or os.path.join(args.model_dir, "txt_export")

    convert_colmap_model_to_txt(args.colmap_exe, args.model_dir, txt_out_dir)
    records = load_points_with_tracks(os.path.join(txt_out_dir, "points3D.txt"))
    records = compute_confidence(records)
    write_confidence_ply(records, args.output)

    counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    for r in records:
        counts[r["classification"]] += 1
    print(f"\nExported {len(records)} points: "
          f"{counts['GREEN']} green, {counts['YELLOW']} yellow, {counts['RED']} red")
    print(f"\nOpen this file in MeshLab (free: https://www.meshlab.net/) "
          f"or any online PLY viewer to view/share it outside this pipeline.")