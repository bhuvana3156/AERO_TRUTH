"""
frame_quality.py
-----------------
PHASE 3 module for ZenX Drone3D (SIH26158).

What this does:
    Looks at every extracted frame and gives it a quality score from 0-100,
    based on measurable image properties. It does NOT select/copy frames
    yet -- that is select_frames.py (the next script). This script only
    measures and reports.

Why this exists:
    A single-pass drone video can have hundreds of frames, but many are
    blurry (motion blur during turns), too dark/bright (sun glare,
    shadows), or near-duplicates of the previous frame (drone moving
    slowly). Feeding all of them into COLMAP wastes time and can hurt
    reconstruction accuracy. We need a measurable way to say "this frame
    is useful" vs "this frame is not," instead of guessing.

Approach: RULE-BASED, not machine learning. Every score is calculated
from a real image statistic -- nothing here is random or hardcoded per
frame. This keeps it simple to explain to judges and easy to debug.

What is measured, per frame:
    1. Sharpness   -- Laplacian variance. Blurry frames have low variance
                      because edges are smeared out. Higher = sharper.
    2. Brightness  -- mean pixel intensity (0-255). Too close to 0 (black)
                      or 255 (white) means the frame carries little usable
                      visual detail.
    3. Overexposure -- % of pixels that are almost pure white (>250).
                        High values mean blown-out highlights (e.g. sun
                        glare), where texture/detail is lost.
    4. Underexposure -- % of pixels that are almost pure black (<5).
                         High values mean crushed shadows with no detail.
    5. Redundancy  -- how similar this frame is to the PREVIOUS kept
                      frame, using color histogram correlation (0 to 1,
                      1 = identical). Near-duplicate frames add no new
                      information for reconstruction.

Final quality_score (0-100) = weighted combination of the above, clipped
to [0, 100]. The exact weights are a simple, explainable starting
heuristic -- not a trained model. They can be tuned later based on
actual reconstruction results (documented as a limitation for judges).

Status: IMPLEMENTED (rule-based scoring only). Frame SELECTION based on
these scores is the next script (select_frames.py) -- not done here.
"""

import cv2
import os
import argparse
import csv
import sys


def compute_sharpness(gray_frame):
    """Laplacian variance: higher = sharper / more in-focus."""
    return cv2.Laplacian(gray_frame, cv2.CV_64F).var()


def compute_brightness(gray_frame):
    """Mean pixel intensity, 0 (black) to 255 (white)."""
    return float(gray_frame.mean())


def compute_exposure_ratios(gray_frame):
    """Fraction of near-white and near-black pixels in the frame."""
    total_pixels = gray_frame.size
    overexposed_ratio = (gray_frame > 250).sum() / total_pixels
    underexposed_ratio = (gray_frame < 5).sum() / total_pixels
    return overexposed_ratio, underexposed_ratio


def compute_histogram(gray_frame):
    """Grayscale histogram, used to compare frames for similarity."""
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def compute_similarity(hist_a, hist_b):
    """
    Correlation between two histograms.
    Returns a value from -1 to 1; 1.0 means the two frames have an
    identical brightness/color distribution (very likely near-duplicates).
    """
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)


def score_frame(sharpness, brightness, overexposed_ratio,
                 underexposed_ratio, similarity_to_prev):
    """
    Combine the raw measurements into a single 0-100 quality score.
    This is a simple weighted rule, explained here so it's easy to
    justify to judges and to tune later:

        - Sharpness contributes up to 50 points.
          (Laplacian variance of 500+ is treated as "very sharp" for a
          typical 720p/1080p frame; this threshold can be re-tuned.)
        - Brightness contributes up to 30 points, penalized the further
          it is from a mid-range value (127 = middle gray).
        - Overexposure/underexposure each subtract up to 15 points.
        - High similarity to the previous kept frame subtracts up to
          20 points (redundant frame -> lower priority to keep).
    """
    sharpness_score = min(50.0, (sharpness / 500.0) * 50.0)

    brightness_distance = abs(brightness - 127.0)  # 0 = perfect mid-gray
    brightness_score = max(0.0, 30.0 - (brightness_distance / 127.0) * 30.0)

    overexposure_penalty = min(15.0, overexposed_ratio * 100.0)
    underexposure_penalty = min(15.0, underexposed_ratio * 100.0)

    redundancy_penalty = 0.0
    if similarity_to_prev is not None and similarity_to_prev > 0.95:
        # Scale penalty: 0.95 similarity -> 0 penalty, 1.0 -> full 20 penalty
        redundancy_penalty = ((similarity_to_prev - 0.95) / 0.05) * 20.0

    raw_score = (sharpness_score + brightness_score
                 - overexposure_penalty - underexposure_penalty
                 - redundancy_penalty)

    return round(max(0.0, min(100.0, raw_score)), 1)


def analyze_frames(frames_dir: str, output_csv: str):
    if not os.path.isdir(frames_dir):
        print(f"[ERROR] Frames folder not found: {frames_dir}")
        sys.exit(1)

    frame_files = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not frame_files:
        print(f"[ERROR] No image files found in: {frames_dir}")
        sys.exit(1)

    results = []
    previous_hist = None

    print(f"Analyzing {len(frame_files)} frames from: {frames_dir}\n")

    for filename in frame_files:
        filepath = os.path.join(frames_dir, filename)
        frame = cv2.imread(filepath)

        if frame is None:
            print(f"[WARNING] Could not read {filename}, skipping.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sharpness = compute_sharpness(gray)
        brightness = compute_brightness(gray)
        overexposed_ratio, underexposed_ratio = compute_exposure_ratios(gray)
        current_hist = compute_histogram(gray)

        similarity_to_prev = None
        if previous_hist is not None:
            similarity_to_prev = compute_similarity(previous_hist, current_hist)

        quality_score = score_frame(
            sharpness, brightness, overexposed_ratio,
            underexposed_ratio, similarity_to_prev
        )

        results.append({
            "filename": filename,
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "overexposed_ratio": round(overexposed_ratio, 4),
            "underexposed_ratio": round(underexposed_ratio, 4),
            "similarity_to_prev": round(similarity_to_prev, 4) if similarity_to_prev is not None else "",
            "quality_score": quality_score,
        })

        print(f"{filename} -> {quality_score}%")

        previous_hist = current_hist

    # Save full results to CSV so the next script (select_frames.py) can use them
    os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "sharpness", "brightness", "overexposed_ratio",
            "underexposed_ratio", "similarity_to_prev", "quality_score"
        ])
        writer.writeheader()
        writer.writerows(results)

    scores = [r["quality_score"] for r in results]
    print(f"\n[DONE] Scored {len(results)} frames.")
    print(f"Score range: {min(scores)}% - {max(scores)}%  (avg: {round(sum(scores)/len(scores), 1)}%)")
    print(f"Full results saved to: {output_csv}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score extracted frames for sharpness, exposure, and redundancy."
    )
    parser.add_argument(
        "--frames", required=True,
        help="Path to folder of extracted frames, e.g. frames/"
    )
    parser.add_argument(
        "--output", default="frames/frame_quality_scores.csv",
        help="Path to save the CSV of scores (default: frames/frame_quality_scores.csv)"
    )

    args = parser.parse_args()
    analyze_frames(args.frames, args.output)