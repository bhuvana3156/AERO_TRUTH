"""
run_pipeline.py
------------------
Master orchestration script for ZenX Drone3D (SIH26158).

What this does:
    Runs the ENTIRE working pipeline -- Phases 1, 3, 4, 9, and the
    model export steps -- with ONE command, in the correct order:

        1. Extract frames from the input video
        2. Score frame quality
        3. Select the best frames (windowed selection)
        4. Run COLMAP sparse reconstruction
        5. Compute the confidence/visibility map
        6. Export the point-cloud .ply model
        7. Export the solid mesh .ply model

Why this exists:
    Running 7 separate commands live in front of judges is risky --
    too many places to mistype something under pressure. This script
    calls each already-tested individual script in sequence (it does
    NOT duplicate their logic -- it just runs them, the same way you
    would by hand), and stops immediately with a clear message if any
    step fails, rather than continuing on broken data.

    Each individual script (extract_frames.py, frame_quality.py, etc.)
    still works standalone too -- this is just a convenience wrapper
    for the full run.

Usage:
    python scripts\\run_pipeline.py --video input\\drone_video\\drone_video.mp4 --colmap-exe "C:\\Tools\\COLMAP\\COLMAP.bat"

Status: IMPLEMENTED (orchestrates existing, individually-tested scripts).
"""

import argparse
import os
import subprocess
import sys


def run_step(step_number, step_name, command_list):
    print(f"\n{'#' * 70}")
    print(f"# STEP {step_number}: {step_name}")
    print(f"# Command: {' '.join(command_list)}")
    print('#' * 70)

    result = subprocess.run(command_list)

    if result.returncode != 0:
        print(f"\n[PIPELINE STOPPED] Step {step_number} ({step_name}) failed "
              f"with exit code {result.returncode}.")
        print("Fix the error above before re-running. No later steps were run, "
              "so no partial/misleading output was produced.")
        sys.exit(1)

    print(f"[OK] Step {step_number} ({step_name}) completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full ZenX Drone3D pipeline with one command."
    )
    parser.add_argument("--video", required=True,
                         help="Path to input video, e.g. input\\drone_video\\drone_video.mp4")
    parser.add_argument("--colmap-exe", default="colmap",
                         help="Path to COLMAP executable, e.g. C:\\Tools\\COLMAP\\COLMAP.bat")
    parser.add_argument("--fps", type=float, default=2.0,
                         help="Frame extraction rate (default: 2.0 frames/sec)")
    parser.add_argument("--window", type=int, default=3,
                         help="Frame selection window size (default: 3)")
    parser.add_argument("--frames-for-reconstruction", choices=["all", "selected"], default="all",
                         help="Use 'all' extracted frames (recommended -- more overlap, better "
                              "chance of success) or 'selected' (fewer, quality-filtered) frames "
                              "for COLMAP reconstruction. Default: all.")
    parser.add_argument("--python-exe", default=sys.executable,
                         help="Python executable to use for sub-steps (default: current one)")

    args = parser.parse_args()
    py = args.python_exe

    print("ZenX Drone3D - Full Pipeline Run")
    print(f"Input video: {args.video}")
    print(f"COLMAP executable: {args.colmap_exe}")

    # STEP 1: Frame extraction
    run_step(1, "Frame Extraction", [
        py, os.path.join("scripts", "extract_frames.py"),
        "--video", args.video,
        "--output", os.path.join("frames", ""),
        "--fps", str(args.fps),
    ])

    # STEP 2: Frame quality scoring
    run_step(2, "Frame Quality Scoring", [
        py, os.path.join("scripts", "frame_quality.py"),
        "--frames", os.path.join("frames", ""),
        "--output", os.path.join("frames", "frame_quality_scores.csv"),
    ])

    # STEP 3: Frame selection
    run_step(3, "Intelligent Frame Selection", [
        py, os.path.join("scripts", "select_frames.py"),
        "--scores", os.path.join("frames", "frame_quality_scores.csv"),
        "--frames", os.path.join("frames", ""),
        "--output", os.path.join("selected_frames", ""),
        "--window", str(args.window),
    ])

    # STEP 4: COLMAP sparse reconstruction
    images_dir = os.path.join("frames", "") if args.frames_for_reconstruction == "all" else os.path.join("selected_frames", "")
    run_step(4, f"COLMAP Sparse Reconstruction (using {images_dir})", [
        py, os.path.join("scripts", "reconstruction.py"),
        "--images", images_dir,
        "--workspace", os.path.join("colmap", ""),
        "--colmap-exe", args.colmap_exe,
    ])

    model_dir = os.path.join("colmap", "sparse", "0")

    # STEP 5: Confidence/visibility map (no interactive window during the run --
    # saved as an image; the final solid model is the demo centerpiece)
    run_step(5, "Confidence/Visibility Map", [
        py, os.path.join("scripts", "confidence_map.py"),
        "--model-dir", model_dir,
        "--colmap-exe", args.colmap_exe,
        "--no-window",
        "--screenshot", os.path.join("output", "confidence_map_preview.png"),
    ])

    # STEP 6: Point cloud .ply export
    run_step(6, "Point Cloud Model Export", [
        py, os.path.join("scripts", "export_model.py"),
        "--model-dir", model_dir,
        "--colmap-exe", args.colmap_exe,
        "--output", os.path.join("output", "zenx_drone3d_model.ply"),
    ])

    # STEP 7: Solid mesh .ply export
    run_step(7, "Solid Mesh Model Export", [
        py, os.path.join("scripts", "generate_solid_mesh.py"),
        "--model-dir", model_dir,
        "--colmap-exe", args.colmap_exe,
        "--output", os.path.join("output", "zenx_drone3d_solid_model.ply"),
    ])

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE -- all 7 steps finished successfully")
    print('=' * 70)
    print("Outputs:")
    print("  frames\\frame_quality_scores.csv       - per-frame quality scores")
    print("  selected_frames\\                       - intelligently selected frames")
    print("  colmap\\sparse\\0\\                       - COLMAP sparse reconstruction")
    print("  output\\confidence_map_preview.png      - confidence map screenshot")
    print("  output\\zenx_drone3d_model.ply          - point cloud 3D model")
    print("  output\\zenx_drone3d_solid_model.ply    - solid mesh 3D model")
    print("\nThese files are your demo deliverables.")


if __name__ == "__main__":
    main()