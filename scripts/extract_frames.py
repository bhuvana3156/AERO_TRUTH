"""
extract_frames.py
------------------
PHASE 1 module for ZenX Drone3D (SIH26158).

What this does:
    Takes a single video file (drone footage OR a mobile test clip) and
    extracts frames from it at a fixed time interval, saving them as
    numbered .jpg images.

Why this exists:
    3D reconstruction tools like COLMAP work on individual images, not on
    video. A drone video shot at 30fps for 60 seconds has 1800 frames --
    far more than we need and far more than COLMAP can process quickly.
    Extracting frames at a controlled rate (e.g. 2 frames per second)
    gives us a manageable, still-overlapping set of images to work with.

    This script does NOT decide which frames are "good" (sharp, useful,
    non-duplicate) -- that is Phase 3 (frame_quality.py / select_frames.py).
    Phase 1 only answers: "how do we turn a video into a folder of images?"

Status: IMPLEMENTED (basic uniform-interval extraction).
Not yet implemented: quality scoring, duplicate/blur rejection, GPS/IMU
metadata extraction from video containers (Phase 3 and Phase 8).
"""

import cv2
import os
import argparse
import sys


def extract_frames(video_path: str, output_dir: str, frames_per_second: float = 2.0):
    """
    Extract frames from a video at a fixed rate.

    Args:
        video_path: path to the input video file (.mp4, .mov, etc.)
        output_dir: folder where extracted frames will be saved
        frames_per_second: how many frames to save per second of video.
                            2.0 means "save 1 frame every 0.5 seconds".

    Returns:
        dict with basic stats about the extraction run.
    """

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Open the video file using OpenCV's video reader
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        print("Possible causes: unsupported codec, corrupted file, or "
              "missing codec support in your OpenCV/ffmpeg install.")
        sys.exit(1)

    # Basic video metadata -- useful for judges' Q&A and for sanity checks
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = total_frames / source_fps if source_fps > 0 else 0

    print("---- Video Info ----")
    print(f"Resolution      : {width}x{height}")
    print(f"Source FPS      : {source_fps:.2f}")
    print(f"Total frames    : {total_frames}")
    print(f"Duration (sec)  : {duration_seconds:.2f}")
    print("---------------------")

    if source_fps <= 0:
        print("[WARNING] Could not read a valid FPS from the video. "
              "Defaulting to 30 fps for interval calculation.")
        source_fps = 30.0

    # How many source frames to skip between each frame we keep.
    # Example: source video is 30fps, we want 2fps output -> skip every 15 frames.
    frame_interval = max(1, round(source_fps / frames_per_second))

    frame_index = 0        # counts every frame read from the video
    saved_count = 0        # counts frames we actually saved

    while True:
        success, frame = cap.read()
        if not success:
            break  # end of video

        if frame_index % frame_interval == 0:
            filename = f"frame_{saved_count:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

        frame_index += 1

    cap.release()

    print(f"[DONE] Extracted {saved_count} frames out of {frame_index} "
          f"total frames read.")
    print(f"Frames saved to: {output_dir}")

    return {
        "source_fps": source_fps,
        "total_source_frames": frame_index,
        "saved_frames": saved_count,
        "output_dir": output_dir,
        "resolution": (width, height),
        "duration_seconds": duration_seconds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from a drone or mobile test video."
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to input video file, e.g. input/drone_video/drone_video.mp4"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output folder for extracted frames, e.g. frames/"
    )
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="Frames to extract per second of video (default: 2.0)"
    )

    args = parser.parse_args()
    extract_frames(args.video, args.output, args.fps)