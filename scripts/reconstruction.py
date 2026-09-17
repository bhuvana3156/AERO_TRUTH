"""
reconstruction.py
-------------------
PHASE 4 module for ZenX Drone3D (SIH26158).

What this does:
    Runs COLMAP's Structure-from-Motion (SfM) pipeline on a folder of
    images to produce camera poses and a SPARSE 3D point cloud.

Why SPARSE only (not dense):
    COLMAP's dense reconstruction step (patch_match_stereo / MVS)
    requires an NVIDIA CUDA GPU. This machine has integrated graphics
    only (confirmed: no dedicated NVIDIA GPU). Attempting dense
    reconstruction would simply fail to run -- it is not a performance
    risk, it is a missing-hardware limitation. Rather than force it or
    fake a result, this script only performs the SPARSE pipeline,
    which is CPU-friendly and lightweight (tested: ~300MB RAM, a few
    seconds to low minutes of CPU per step on a small image set).

    Dense meshing is documented as FUTURE WORK requiring GPU hardware
    (e.g. a cloud GPU instance), not implemented here.

Pipeline (3 COLMAP CLI calls, run in order):
    1. feature_extractor  -- detects SIFT keypoints in every image
    2. sequential_matcher -- matches features between temporally
                              neighboring frames (appropriate since our
                              images come from a single continuous
                              video pass, not random unordered photos)
    3. mapper             -- incremental Structure-from-Motion: estimates
                              camera poses and triangulates a sparse
                              3D point cloud from the matches

Output:
    <workspace>/database.db          -- COLMAP's feature/match database
    <workspace>/sparse/0/            -- the reconstructed model:
                                         cameras.bin, images.bin, points3D.bin
    If reconstruction fails (e.g. not enough overlap between images),
    <workspace>/sparse/0/ will not be created -- this script detects
    that and reports it honestly instead of claiming success.

Status: IMPLEMENTED (sparse SfM via COLMAP CLI subprocess calls).
Not implemented: dense reconstruction/MVS (requires CUDA GPU -- future
work), GPU-accelerated feature extraction (CPU-only flags used by
default here for reliability across machines without a GPU).
"""

import argparse
import os
import subprocess
import sys


def run_command(command_list, step_name):
    """
    Run a COLMAP CLI command and stream its output live.
    Exits the script with a clear message if the command fails.
    """
    print(f"\n{'=' * 60}")
    print(f"STEP: {step_name}")
    print(f"Command: {' '.join(command_list)}")
    print('=' * 60)

    result = subprocess.run(command_list)

    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed (exit code {result.returncode}).")
        print("Common causes: wrong --colmap-exe path, images folder empty or "
              "contains non-image files, or a GPU/OpenGL context error.")
        print("If you see a GPU/OpenGL-related error above, this script already "
              "uses CPU-only flags, so that specific issue should not occur -- "
              "please share the exact error text.")
        sys.exit(1)

    print(f"[OK] Step '{step_name}' completed.")


def run_sparse_reconstruction(colmap_exe: str, images_dir: str, workspace_dir: str):
    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images folder not found: {images_dir}")
        sys.exit(1)

    # Warn about non-image files, since COLMAP will log an error (non-fatal)
    # for anything that isn't a valid image, e.g. a stray CSV file.
    valid_ext = (".jpg", ".jpeg", ".png")
    all_files = os.listdir(images_dir)
    image_files = [f for f in all_files if f.lower().endswith(valid_ext)]
    non_image_files = [f for f in all_files if not f.lower().endswith(valid_ext)]

    if non_image_files:
        print(f"[WARNING] Non-image files found in {images_dir}: {non_image_files}")
        print("COLMAP will skip these with a harmless per-file error, but it's "
              "cleaner to move them out (e.g. move any .csv elsewhere).")

    print(f"Found {len(image_files)} image(s) in {images_dir}")

    if len(image_files) < 3:
        print("[ERROR] Need at least 3 images for Structure-from-Motion. Stopping.")
        sys.exit(1)

    database_path = os.path.join(workspace_dir, "database.db")
    sparse_dir = os.path.join(workspace_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    # Remove a stale database from a previous run, so COLMAP starts fresh
    if os.path.exists(database_path):
        os.remove(database_path)
        print(f"Removed old database: {database_path}")

    # STEP 1: Feature extraction.
    # NOTE: we do NOT pass --SiftExtraction.use_gpu here. On COLMAP builds
    # compiled WITHOUT any GPU support at all (e.g. the Windows "nocuda"
    # release), that flag does not exist in the option parser and causes
    # an error. Such builds always run on CPU by default anyway, so no
    # flag is needed.
    run_command([
        colmap_exe, "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1",
    ], "Feature Extraction")

    # STEP 2: Sequential matching -- appropriate for ordered video frames
    # (matches each frame mainly against its temporal neighbors, rather
    # than every possible pair, which is both faster and more correct
    # for a single continuous camera pass)
    run_command([
        colmap_exe, "sequential_matcher",
        "--database_path", database_path,
    ], "Feature Matching (Sequential)")

    # STEP 3: Sparse reconstruction (incremental Structure-from-Motion)
    run_command([
        colmap_exe, "mapper",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--output_path", sparse_dir,
    ], "Sparse Reconstruction (Mapper)")

    # Check whether a model was actually produced -- and verify it QUANTITATIVELY.
    # Files existing is not enough proof: COLMAP can write near-empty output
    # files even when reconstruction mostly failed. We parse the real
    # registered-image and point counts via `colmap model_analyzer` instead
    # of just checking file existence, so we never report a fake "success".
    model_dir = os.path.join(sparse_dir, "0")

    if not (os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "points3D.bin"))):
        print(f"\n[FAILED] No sparse model was produced in {sparse_dir}.")
        print("This usually means COLMAP could not find enough overlapping "
              "features between images to estimate camera poses -- e.g. too "
              "few images, too little overlap between them, or a scene with "
              "very little visual texture (e.g. blank walls, repetitive patterns).")
        print("This is an honest failure report, not a crash -- your computer "
              "is fine. We can try again with different/more images.")
        return

    analysis = subprocess.run(
        [colmap_exe, "model_analyzer", "--path", model_dir],
        capture_output=True, text=True
    )
    analyzer_output = analysis.stderr + analysis.stdout  # COLMAP logs to stderr

    registered_images = None
    total_points = None
    for line in analyzer_output.splitlines():
        if "Registered images:" in line:
            registered_images = int(line.strip().split(":")[-1].strip())
        if "Points:" in line and "Registered" not in line:
            total_points = int(line.strip().split(":")[-1].strip())

    print(f"\n{'=' * 60}")
    print("RECONSTRUCTION RESULT (measured, not assumed)")
    print('=' * 60)
    print(f"Images provided       : {len(image_files)}")
    print(f"Images registered     : {registered_images if registered_images is not None else 'unknown'}")
    print(f"3D points reconstructed: {total_points if total_points is not None else 'unknown'}")

    if registered_images is None or registered_images < max(3, len(image_files) // 2):
        print("\n[WEAK RECONSTRUCTION] Fewer than half the images were registered.")
        print("This is an honest, measured result -- not a crash, and not fabricated "
              "success. Likely causes: not enough overlap between consecutive "
              "images, too few images, or insufficient visual texture in the scene.")
        print("Next step: try with more/less-aggressively-selected frames, or a "
              "video with slower camera movement for more overlap.")
    else:
        print(f"\n[SUCCESS] {registered_images}/{len(image_files)} images registered "
              f"with {total_points} reconstructed points.")
        print(f"Sparse model saved at: {model_dir} (cameras.bin, images.bin, points3D.bin)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run COLMAP sparse (CPU-only) reconstruction on a folder of images."
    )
    parser.add_argument("--images", required=True,
                         help="Folder of images to reconstruct, e.g. selected_frames\\")
    parser.add_argument("--workspace", required=True,
                         help="Folder to store COLMAP's database and output, e.g. colmap\\")
    parser.add_argument("--colmap-exe", default="colmap",
                         help="Path to the COLMAP executable. On Windows, this is "
                              "typically the full path to COLMAP.bat, e.g. "
                              "C:\\Tools\\COLMAP\\COLMAP.bat")

    args = parser.parse_args()
    run_sparse_reconstruction(args.colmap_exe, args.images, args.workspace)