"""
select_frames.py
------------------
PHASE 3 module for ZenX Drone3D (SIH26158) -- continued.

What this does:
    Reads the quality scores produced by frame_quality.py and copies a
    subset of "selected" frames into selected_frames/, ready for COLMAP.

Why NOT simply keep the globally highest-scoring frames:
    A single-pass drone/phone video usually has quality drift -- e.g.
    motion blur increasing as the camera speeds up partway through.
    If we just sorted all frames by score and kept the top N, we would
    end up keeping mostly frames from ONE part of the flight path and
    dropping the rest entirely. That leaves a gap in camera coverage,
    and COLMAP cannot reconstruct parts of the scene it never has a
    good view of -- no matter how sharp the frames we did keep are.

Approach used here: WINDOWED best-frame selection.
    The frame sequence is split into consecutive windows of a fixed
    size (default: 3 frames per window). From each window, we keep
    ONLY the single frame with the highest quality_score.
    This guarantees:
        - Frames are kept from across the ENTIRE video, preserving
          camera-path coverage for reconstruction.
        - Within each small window, we still discard the blurriest/
          worst-exposed/most redundant frame.
    A minimum quality floor can optionally be set (--min-quality) to
    drop a window's best frame anyway if even the best one is too poor
    to use -- disabled by default so we never leave a coverage gap
    silently.

Status: IMPLEMENTED (rule-based, windowed selection).
Not yet implemented: any learning-based or geometry-aware selection
(e.g. using estimated camera motion/parallax instead of a fixed window
size) -- documented as a possible future improvement.
"""

import os
import csv
import shutil
import argparse
import sys


def load_scores(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"[ERROR] Score CSV not found: {csv_path}")
        print("Run frame_quality.py first to generate it.")
        sys.exit(1)

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["quality_score"] = float(row["quality_score"])
            rows.append(row)

    if not rows:
        print(f"[ERROR] No rows found in: {csv_path}")
        sys.exit(1)

    return rows


def select_frames(rows, window_size: int, min_quality: float):
    """
    Split rows (already in original frame order) into windows of
    `window_size` consecutive frames, and keep the best-scoring frame
    from each window. Returns the list of selected rows, in order.
    """
    selected = []
    dropped_low_quality_windows = 0

    for start in range(0, len(rows), window_size):
        window = rows[start:start + window_size]
        best_in_window = max(window, key=lambda r: r["quality_score"])

        if best_in_window["quality_score"] < min_quality:
            dropped_low_quality_windows += 1
            continue

        selected.append(best_in_window)

    return selected, dropped_low_quality_windows


def copy_selected_frames(selected_rows, frames_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for i, row in enumerate(selected_rows):
        src = os.path.join(frames_dir, row["filename"])
        dst_filename = f"selected_{i:04d}.jpg"
        dst = os.path.join(output_dir, dst_filename)

        if not os.path.exists(src):
            print(f"[WARNING] Source frame missing, skipping: {src}")
            continue

        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Select the best frame from each window using frame_quality.py's scores."
    )
    parser.add_argument("--scores", required=True,
                         help="Path to frame_quality_scores.csv")
    parser.add_argument("--frames", required=True,
                         help="Path to the folder containing the original extracted frames")
    parser.add_argument("--output", required=True,
                         help="Path to save selected frames, e.g. selected_frames/")
    parser.add_argument("--window", type=int, default=3,
                         help="Number of consecutive frames per window (default: 3). "
                              "The single best-scoring frame in each window is kept.")
    parser.add_argument("--min-quality", type=float, default=0.0,
                         help="Optional floor (0-100). A window's best frame is still "
                              "dropped if it scores below this (default: 0 = never drop, "
                              "to avoid coverage gaps).")

    args = parser.parse_args()

    rows = load_scores(args.scores)
    selected_rows, dropped_windows = select_frames(rows, args.window, args.min_quality)

    if not selected_rows:
        print("[ERROR] No frames were selected. Lower --min-quality or check your data.")
        sys.exit(1)

    copy_selected_frames(selected_rows, args.frames, args.output)

    total_windows = (len(rows) + args.window - 1) // args.window
    scores = [r["quality_score"] for r in selected_rows]

    print(f"Total frames analyzed : {len(rows)}")
    print(f"Window size           : {args.window} frames/window")
    print(f"Total windows         : {total_windows}")
    print(f"Windows dropped (below min-quality {args.min_quality}%) : {dropped_windows}")
    print(f"Frames selected       : {len(selected_rows)}")
    print(f"Selected score range  : {min(scores):.1f}% - {max(scores):.1f}% "
          f"(avg: {sum(scores)/len(scores):.1f}%)")
    print(f"Selected frames saved to: {args.output}")
    print()
    print("Selected frames (original -> renamed):")
    for i, row in enumerate(selected_rows):
        print(f"  {row['filename']} (score {row['quality_score']:.1f}%) -> selected_{i:04d}.jpg")


if __name__ == "__main__":
    main()