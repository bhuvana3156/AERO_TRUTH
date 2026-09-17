"""
confidence_map.py
-------------------
PHASE 9 module for ZenX Drone3D (SIH26158).

What this does:
    For every reconstructed 3D point, computes a measurable confidence
    score and classifies it as GREEN (high confidence), YELLOW (medium),
    or RED (low/uncertain) -- then visualizes the point cloud colored
    by this classification instead of by original image color.

Why this matters (this is the project's key differentiator):
    A plain "video -> COLMAP -> 3D model" pipeline treats every
    reconstructed point as equally trustworthy. In reality, some points
    were seen by many camera views from multiple angles and triangulated
    precisely; others were barely observed (e.g. only the minimum 2
    views) or triangulated with high error, meaning they are much less
    reliable -- especially likely in a SINGLE-PASS video, where many
    surfaces are only seen briefly from one direction. Surfacing this
    honestly is more useful (and more novel) than hiding it.

What "confidence" is measured from (real data COLMAP already gives us,
NOT randomly generated):
    1. Track length -- how many different camera images observed this
       exact 3D point. More independent observations = more reliable
       triangulation. Minimum possible is 2 (you need at least 2 views
       to triangulate a point at all).
    2. Reprojection error (pixels) -- COLMAP's own measure of how well
       the triangulated 3D point re-projects back into each image that
       observed it. Lower error = more geometrically consistent = more
       trustworthy.

Confidence score formula (0-100, simple and explainable -- NOT a
trained model):
    - Track length contributes up to 70 points. A track length of 2
      (bare minimum) scores 0 on this component; a track length of 8+
      scores the full 70. Linearly scaled in between.
    - Reprojection error contributes up to 30 points. 0px error scores
      the full 30; error of 2px or worse scores 0. Linearly scaled
      in between.
    - Final score = sum of the two, clipped to [0, 100].

Classification thresholds (also simple and explainable, tunable later
based on real project needs):
    GREEN  (score >= 70): well-observed, low-error -- high confidence
    YELLOW (40 <= score < 70): partially observed / moderate error
    RED    (score < 40): barely observed and/or high error -- treat
                          as uncertain, likely near an occlusion edge
                          or a weak part of the single-pass coverage

Status: IMPLEMENTED (rule-based, using COLMAP's own real per-point
track-length and reprojection-error data).
Not yet implemented: incorporating depth-map consistency (would need
dense/MVS reconstruction, which requires a CUDA GPU this machine does
not have) or view-angle spread (would need per-observation ray
directions, a possible future refinement). Documented as future work.
"""

import argparse
import os
import sys
import subprocess
import csv

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
    """
    Parses points3D.txt and extracts, per point:
        x, y, z, reprojection_error, track_length
    (track_length = number of (IMAGE_ID, POINT2D_IDX) pairs listed)
    """
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

    print(f"Loaded {len(records)} points with track/error data")
    if not records:
        print("[ERROR] No points found.")
        sys.exit(1)

    return records


def compute_confidence(records, max_track_for_full_score: int = 8,
                        max_error_for_zero_score: float = 2.0):
    """
    Adds 'confidence_score' (0-100) and 'classification'
    ('GREEN' / 'YELLOW' / 'RED') to each record, using the formula
    documented at the top of this file.
    """
    for r in records:
        # Track length component: up to 70 points
        track_component = min(
            70.0,
            max(0.0, (r["track_length"] - 2) / (max_track_for_full_score - 2)) * 70.0
        )

        # Reprojection error component: up to 30 points
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


def save_confidence_csv(records, output_csv: str):
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "x", "y", "z", "error", "track_length",
            "confidence_score", "classification"
        ])
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved per-point confidence data to: {output_csv}")


def print_summary(records):
    total = len(records)
    counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    for r in records:
        counts[r["classification"]] += 1

    print(f"\n{'=' * 50}")
    print("CONFIDENCE / VISIBILITY SUMMARY (measured, not assumed)")
    print('=' * 50)
    for label in ("GREEN", "YELLOW", "RED"):
        pct = counts[label] / total * 100
        print(f"{label:7s}: {counts[label]:5d} points ({pct:.1f}%)")
    print(f"Total points analyzed: {total}")

    avg_track = np.mean([r["track_length"] for r in records])
    avg_error = np.mean([r["error"] for r in records])
    print(f"\nAverage track length : {avg_track:.2f} views/point")
    print(f"Average reprojection error: {avg_error:.3f} px")


def plot_confidence_map(records, screenshot_path: str = None, show_window: bool = True):
    color_map = {"GREEN": (0.1, 0.7, 0.1), "YELLOW": (0.9, 0.8, 0.0), "RED": (0.85, 0.1, 0.1)}

    xyz = np.array([[r["x"], r["y"], r["z"]] for r in records])
    colors = np.array([color_map[r["classification"]] for r in records])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=3)
    ax.set_title("ZenX Drone3D - Confidence/Visibility Map (Phase 9)\n"
                  "Green = high confidence, Yellow = medium, Red = uncertain")
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
        print(f"Saved confidence map image to: {screenshot_path}")

    if show_window:
        print("\nOpening interactive confidence map viewer...")
        print("Controls: left-click+drag to rotate, scroll to zoom.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and visualize a confidence/visibility map for a COLMAP sparse model."
    )
    parser.add_argument("--model-dir", required=True,
                         help="Path to COLMAP sparse model folder, e.g. colmap\\sparse\\0")
    parser.add_argument("--colmap-exe", default="colmap",
                         help="Path to COLMAP executable")
    parser.add_argument("--txt-out-dir", default=None,
                         help="Where to save converted TXT files (default: <model-dir>\\txt_export)")
    parser.add_argument("--output-csv", default=None,
                         help="Where to save per-point confidence CSV "
                              "(default: <model-dir>\\confidence_scores.csv)")
    parser.add_argument("--no-window", action="store_true",
                         help="Skip the interactive window, just compute + save")
    parser.add_argument("--screenshot", default=None,
                         help="Optional path to save a static preview PNG")

    args = parser.parse_args()

    txt_out_dir = args.txt_out_dir or os.path.join(args.model_dir, "txt_export")
    output_csv = args.output_csv or os.path.join(args.model_dir, "confidence_scores.csv")

    convert_colmap_model_to_txt(args.colmap_exe, args.model_dir, txt_out_dir)
    records = load_points_with_tracks(os.path.join(txt_out_dir, "points3D.txt"))
    records = compute_confidence(records)
    save_confidence_csv(records, output_csv)
    print_summary(records)

    plot_confidence_map(records, screenshot_path=args.screenshot, show_window=not args.no_window)