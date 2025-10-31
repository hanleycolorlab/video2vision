#!/usr/bin/env python3
"""
Step 1b: Run automatic alignments using saved flip configs.

This reads the flip configurations you created in Step 1a and runs
AutoAlign on all video pairs to calculate alignment parameters.

By default:
- Main videos use ECC alignment (slow but robust, no markers needed)
- Calibration videos use ArUco alignment (fast, requires markers)

Usage (from project root):
    # Using module syntax:
    python -m scripts.step1b_run_alignments --all --save-preview
    python -m scripts.step1b_run_alignments \
        --samples 001 006 012 --save-preview

    # Or if video2vision is installed:
    python scripts/step1b_run_alignments.py --all --save-preview
    python scripts/step1b_run_alignments.py \
        --samples 001 006 012 --save-preview

    # GENERATE PREVIEWS ONLY: If you forgot --save-preview,
    # regenerate previews from existing alignments
    python scripts/step1b_run_alignments.py --all --preview-only

    # Override methods if needed
    python scripts/step1b_run_alignments.py \
        --all --main-method any --calibration-method aruco

    # Use simpler motion model (rotation + translation only, no skew/warp)
    python scripts/step1b_run_alignments.py \
        --all --save-preview --main-motion-type euclidean

    # Use affine motion model (rotation + translation + scale + shear,
    # but no perspective warp)
    python scripts/step1b_run_alignments.py \
        --all --save-preview --main-motion-type affine

    # Reprocess only samples that were rejected in step1c review
    python scripts/step1b_run_alignments.py \
        --all --rejected-only --save-preview --main-motion-type euclidean

    # Disable temporal alignment (if cameras are perfectly synced)
    python scripts/step1b_run_alignments.py \
        --all --save-preview --no-temporal

    # Process only main videos or only calibration videos
    python scripts/step1b_run_alignments.py \
        --all --main-only --save-preview
    python scripts/step1b_run_alignments.py \
        --all --calibration-only --save-preview

    # Use a known-good alignment as starting point for other samples
    python scripts/step1b_run_alignments.py \
        --samples 001 --save-preview --save-as-template
    python scripts/step1b_run_alignments.py \
        --all --save-preview --use-initial-transform 001

    # Or manually add initial_transform to config.json and
    # it will be used automatically:
    # {
    #   "alignment_main": {
    #     "initial_transform": [[0.998, -0.052, 45.2],
    #                          [0.051, 0.997, -12.3], [0, 0, 1]]
    #   }
    # }

Recommended workflow:
    1. Run with --save-preview to calculate alignment params and save
       preview images (~30 sec per sample with ArUco, 1-2 min with ECC)
    2. Run step1c to review the composite images and
       approve/reject alignments
    3. If some alignments were rejected, reprocess with --rejected-only
       and different settings (e.g., --main-motion-type euclidean)
    4. Repeat step1c review until all alignments are approved
    5. If you forgot --save-preview, use --preview-only to quickly
       regenerate previews from saved alignments
    6. Once all verified, use step2 to apply those saved alignments to
       full videos (overnight)
"""

import argparse
import json
import sys
import time
from pathlib import Path

from video2vision import io, pipeline
from video2vision.auto_operators import (
    AutoAlign, AutoTemporalAlign, AlignmentNotFound
)
from video2vision.operators import HorizontalFlip, VerticalFlip


def load_sample_config(sample_id, samples_dir="videos/samples"):
    """Load configuration for a sample from its folder"""
    config_path = Path(samples_dir) / sample_id / "config.json"

    if not config_path.exists():
        return None

    with open(config_path, "r") as f:
        return json.load(f)


def find_video_pair(sample_dir, use_calibration=False):
    """Find VIS and UV video pair"""
    sample_dir = Path(sample_dir)

    if use_calibration:
        sample_dir = sample_dir / "calibration"

    if not sample_dir.exists():
        return None, None

    vis_videos = sorted(sample_dir.glob("VIS_*.MP4")) + sorted(
        sample_dir.glob("VIS_*.mp4")
    )
    uv_videos = sorted(sample_dir.glob("UV_*.MP4")) + sorted(
        sample_dir.glob("UV_*.mp4")
    )

    if not vis_videos or not uv_videos:
        return None, None

    return str(vis_videos[0]), str(uv_videos[0])


def _trim_video(input_path, output_path, max_frames):
    """
    Create a temporary trimmed version of a video with only the first
    max_frames frames
    """
    import cv2

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()


def generate_preview_from_alignment(
    vis_path,
    uv_path,
    flip_type,
    homography_matrix,
    output_size,
    preview_dir,
    sample_id,
    video_type="main",
):
    """Generate preview images using existing alignment parameters"""
    import cv2
    import numpy as np
    from video2vision.warp import Warp

    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Delete existing preview files
    for old_file in preview_dir.glob(f"{sample_id}_{video_type}_*.png"):
        old_file.unlink()

    # Read first frame from each video
    vis_cap = cv2.VideoCapture(vis_path)
    uv_cap = cv2.VideoCapture(uv_path)

    ret_vis, vis_frame = vis_cap.read()
    ret_uv, uv_frame = uv_cap.read()

    vis_cap.release()
    uv_cap.release()

    if not ret_vis or not ret_uv:
        print("    ⚠ Could not read frames from videos")
        return

    # Convert to float32 [0, 1] range
    vis_img = vis_frame.astype(np.float32) / 255.0
    uv_img = uv_frame.astype(np.float32) / 255.0

    # Apply flip to UV if needed
    if flip_type == "horizontal":
        uv_img = np.flip(uv_img, axis=1)
    elif flip_type == "vertical":
        uv_img = np.flip(uv_img, axis=0)

    # Apply alignment to UV
    if homography_matrix is not None:
        homography_array = np.array(homography_matrix)
        warp_op = Warp(homography_array, output_size=tuple(output_size))
        uv_aligned_dict = warp_op.apply({"image": uv_img})
        uv_aligned = uv_aligned_dict["image"]
    else:
        uv_aligned = uv_img

    # Convert back to uint8 for saving
    vis_uint8 = (np.clip(vis_img, 0, 1) * 255).astype(np.uint8)
    uv_aligned_uint8 = (np.clip(uv_aligned, 0, 1) * 255).astype(np.uint8)

    # Create composite (50% opacity overlay)
    composite = cv2.addWeighted(
        vis_uint8, 0.5, uv_aligned_uint8, 0.5, 0
    )

    # Save all three images
    vis_name = f"{sample_id}_{video_type}_vis.png"
    uv_name = f"{sample_id}_{video_type}_uv_aligned.png"
    composite_name = f"{sample_id}_{video_type}_composite.png"
    vis_path_out = preview_dir / vis_name
    uv_path_out = preview_dir / uv_name
    composite_path_out = preview_dir / composite_name

    cv2.imwrite(str(vis_path_out), vis_uint8)
    cv2.imwrite(str(uv_path_out), uv_aligned_uint8)
    cv2.imwrite(str(composite_path_out), composite)

    print("    Previews generated:")
    print(f"      - {sample_id}_{video_type}_vis.png")
    print(f"      - {sample_id}_{video_type}_uv_aligned.png")
    print(f"      - {sample_id}_{video_type}_composite.png")


def run_alignment(
    vis_path,
    uv_path,
    flip_type,
    method="any",
    temporal=True,
    save_preview=False,
    preview_dir=None,
    sample_id=None,
    video_type="main",
    motion_type="homography",
    initial_transform=None,
):
    """Run AutoAlign and return the fitted operator with alignment parameters

    Note: AutoAlign/AutoTemporalAlign automatically only use the first
    batch (30 frames) to calculate alignment parameters.

    Args:
        save_preview: If True, save 3 preview images (UV aligned, VIS,
                      composite)
        preview_dir: Directory to save preview images
        sample_id: Sample ID for naming preview files
        video_type: "main" or "calibration" for naming preview files

    Returns:
        fitted_align_op: The fitted alignment operator with all parameters
    """
    import cv2
    import numpy as np

    # Get video dimensions and frame counts
    vis_cap = cv2.VideoCapture(vis_path)
    width = int(vis_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vis_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vis_frame_count = int(vis_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vis_cap.release()

    uv_cap = cv2.VideoCapture(uv_path)
    uv_frame_count = int(uv_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    uv_cap.release()

    expected_size = (width, height)

    # Use minimum frame count to handle mismatched video lengths
    min_frame_count = min(vis_frame_count, uv_frame_count)
    # Use up to 30 frames, but not more than available
    batch_size = min(30, min_frame_count)

    if vis_frame_count != uv_frame_count:
        msg = f"VIS={vis_frame_count}, UV={uv_frame_count}"
        print(f"    ⚠ Frame count mismatch: {msg}")
        print(f"    Will use first {min_frame_count} frames for alignment")

    # Build pipeline for alignment calculation
    pipe = pipeline.Pipeline()

    # Create temporary trimmed videos if there's a frame count mismatch
    # This ensures both loaders produce exactly the same number of frames
    import tempfile
    temp_vis_path = vis_path
    temp_uv_path = uv_path
    temp_files_to_cleanup = []

    if vis_frame_count != uv_frame_count:
        # Need to trim the longer video to match the shorter one
        temp_dir = Path(tempfile.gettempdir())

        if vis_frame_count > min_frame_count:
            # Trim VIS video
            vis_stem = Path(vis_path).stem
            temp_vis_path = temp_dir / f"temp_vis_{vis_stem}_trimmed.mp4"
            _trim_video(vis_path, str(temp_vis_path), min_frame_count)
            temp_files_to_cleanup.append(temp_vis_path)

        if uv_frame_count > min_frame_count:
            # Trim UV video
            uv_stem = Path(uv_path).stem
            temp_uv_path = temp_dir / f"temp_uv_{uv_stem}_trimmed.mp4"
            _trim_video(uv_path, str(temp_uv_path), min_frame_count)
            temp_files_to_cleanup.append(temp_uv_path)

    # Loaders
    vis_loader_idx = pipe.add_operator(
        io.Loader(
            str(temp_vis_path),
            expected_size=expected_size,
            batch_size=batch_size
        )
    )
    uv_loader_idx = pipe.add_operator(
        io.Loader(
            str(temp_uv_path),
            expected_size=expected_size,
            batch_size=batch_size
        )
    )

    # Apply flip to UV if needed
    current_idx = uv_loader_idx
    if flip_type == "horizontal":
        flip_idx = pipe.add_operator(HorizontalFlip())
        pipe.add_edge(uv_loader_idx, flip_idx, in_slot=0)
        current_idx = flip_idx
    elif flip_type == "vertical":
        flip_idx = pipe.add_operator(VerticalFlip())
        pipe.add_edge(uv_loader_idx, flip_idx, in_slot=0)
        current_idx = flip_idx

    # Auto alignment
    if temporal:
        align_op = AutoTemporalAlign(
            time_shift_range=(-10, 10),
            method=method,
            num_votes=3,
            bands=[[0, 1, 2], []],
            motion_type=motion_type,
            initial_transform=initial_transform,
        )
    else:
        align_op = AutoAlign(
            method=method,
            num_votes=3,
            bands=[[0, 1, 2], []],
            motion_type=motion_type,
            initial_transform=initial_transform,
        )

    align_idx = pipe.add_operator(align_op)
    # UV to slot 0 (source)
    pipe.add_edge(current_idx, align_idx, in_slot=0)
    # VIS to slot 1 (control)
    pipe.add_edge(vis_loader_idx, align_idx, in_slot=1)

    # Temporary writer (need one to run the pipeline)
    import tempfile

    temp_output = Path(tempfile.mktemp(suffix=".mp4", prefix="align_"))
    writer_idx = pipe.add_operator(io.Writer(str(temp_output)))
    pipe.add_edge(align_idx, writer_idx, in_slot=0)

    # Run pipeline to calculate alignment
    try:
        pipe.run()
    except Exception as e:
        raise e
    finally:
        # Clean up temp files
        if temp_output.exists():
            temp_output.unlink()
        for temp_file in temp_files_to_cleanup:
            if temp_file.exists():
                temp_file.unlink()

    # Extract fitted alignment operator
    fitted_align_op = pipe.nodes[align_idx]["operator"]

    # Store frame count info on the operator for later reference
    fitted_align_op.vis_frame_count = vis_frame_count
    fitted_align_op.uv_frame_count = uv_frame_count
    fitted_align_op.usable_frame_count = min_frame_count

    # Generate preview images if requested
    if save_preview and preview_dir and sample_id:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

        # Delete existing preview files to avoid FileExistsError
        for old_file in preview_dir.glob(f"{sample_id}_{video_type}_*.png"):
            old_file.unlink()

        # Read first frame from each video
        vis_cap = cv2.VideoCapture(vis_path)
        uv_cap = cv2.VideoCapture(uv_path)

        ret_vis, vis_frame = vis_cap.read()
        ret_uv, uv_frame = uv_cap.read()

        vis_cap.release()
        uv_cap.release()

        if ret_vis and ret_uv:
            # Convert to float32 [0, 1] range (video2vision format)
            vis_img = vis_frame.astype(np.float32) / 255.0
            uv_img = uv_frame.astype(np.float32) / 255.0

            # Apply flip to UV if needed
            if flip_type == "horizontal":
                uv_img = np.flip(uv_img, axis=1)
            elif flip_type == "vertical":
                uv_img = np.flip(uv_img, axis=0)

            # Apply alignment to UV
            # Warp operator expects dict with 'image' key
            from video2vision.warp import Warp

            has_coe = (hasattr(fitted_align_op, "coe") and
                       fitted_align_op.coe is not None)
            if has_coe:
                warp_op = Warp(fitted_align_op.coe, output_size=expected_size)
                uv_aligned_dict = warp_op.apply({"image": uv_img})
                uv_aligned = uv_aligned_dict["image"]
            else:
                uv_aligned = uv_img

            # Convert back to uint8 for saving
            vis_uint8 = (np.clip(vis_img, 0, 1) * 255).astype(np.uint8)
            uv_clipped = np.clip(uv_aligned, 0, 1)
            uv_aligned_uint8 = (uv_clipped * 255).astype(np.uint8)

            # Create composite (50% opacity overlay)
            composite = cv2.addWeighted(
                vis_uint8, 0.5, uv_aligned_uint8, 0.5, 0
            )

            # Save all three images
            vis_name = f"{sample_id}_{video_type}_vis.png"
            uv_name = f"{sample_id}_{video_type}_uv_aligned.png"
            composite_name = f"{sample_id}_{video_type}_composite.png"
            vis_path_out = preview_dir / vis_name
            uv_path_out = preview_dir / uv_name
            composite_path_out = preview_dir / composite_name

            cv2.imwrite(str(vis_path_out), vis_uint8)
            cv2.imwrite(str(uv_path_out), uv_aligned_uint8)
            cv2.imwrite(str(composite_path_out), composite)

    return fitted_align_op


def main():
    parser = argparse.ArgumentParser(
        description="Run automatic alignments using saved flip configs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all samples with flip configs"
    )
    group.add_argument("--samples", nargs="+", help="Specific samples")

    parser.add_argument(
        "--main-method",
        default="ecc",
        choices=["any", "aruco", "ecc"],
        help="Alignment method for main videos (default: ecc)",
    )
    parser.add_argument(
        "--calibration-method",
        default="aruco",
        choices=["any", "aruco", "ecc"],
        help="Alignment method for calibration videos (default: aruco)",
    )
    parser.add_argument(
        "--main-motion-type",
        default="homography",
        choices=["euclidean", "affine", "homography"],
        help="Motion model for main videos (default: homography). "
        "euclidean=rotation+translation only (no skew/warp), "
        "affine=rotation+translation+scale+shear, "
        "homography=full perspective transform with skew/warp",
    )
    parser.add_argument(
        "--calibration-motion-type",
        default="homography",
        choices=["euclidean", "affine", "homography"],
        help="Motion model for calibration videos (default: homography)",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration videos (only process main videos)",
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Only process main videos (same as --skip-calibration)",
    )
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Only process calibration videos (skip main videos)",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Disable temporal alignment for main videos "
             "(temporal_shift will be 0)",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save preview images of aligned first frame for verification",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only generate preview images from existing alignments "
             "(skip alignment calculation)",
    )
    parser.add_argument(
        "--rejected-only",
        action="store_true",
        help="Only process samples with rejected alignments "
             "(from step1c review)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if alignment already exists",
    )

    parser.add_argument(
        "--samples-dir",
        default="videos/samples",
        help="Directory containing samples (default: videos/samples)",
    )
    parser.add_argument(
        "--use-initial-transform",
        metavar="SAMPLE_ID",
        help="Use the alignment from SAMPLE_ID as the initial transform "
             "for ECC optimization. This provides a starting point for "
             "the alignment, improving robustness and speed.",
    )
    parser.add_argument(
        "--save-as-template",
        action="store_true",
        help="After successful alignment, save the transform as "
             "'initial_transform' in config so it can be reused for "
             "future alignments or copied to other samples.",
    )

    args = parser.parse_args()

    # Validate conflicting flags
    if args.main_only and args.calibration_only:
        print("Error: Cannot use both --main-only and --calibration-only")
        sys.exit(1)

    # Normalize flags: --main-only is same as --skip-calibration
    if args.main_only:
        args.skip_calibration = True

    # Setup samples directory
    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        sys.exit(1)

    # Load initial transform if specified
    initial_transform_main = None
    initial_transform_cal = None
    if args.use_initial_transform:
        template_config = load_sample_config(
            args.use_initial_transform, args.samples_dir
        )
        if template_config is None:
            template_id = args.use_initial_transform
            print(f"Error: Template sample '{template_id}' config not found")
            sys.exit(1)

        # Load main video initial transform
        alignment_main = template_config.get('alignment_main', {})
        if alignment_main.get('homography_matrix'):
            initial_transform_main = alignment_main['homography_matrix']
            template_id = args.use_initial_transform
            print(f"Loaded initial transform from sample "
                  f"'{template_id}' (main)")
        elif alignment_main.get('initial_transform'):
            initial_transform_main = alignment_main['initial_transform']
            template_id = args.use_initial_transform
            print(f"Loaded initial transform template from sample "
                  f"'{template_id}' (main)")

        # Load calibration video initial transform
        alignment_cal = template_config.get('alignment_calibration', {})
        if alignment_cal.get('homography_matrix'):
            initial_transform_cal = alignment_cal['homography_matrix']
            template_id = args.use_initial_transform
            print(f"Loaded initial transform from sample "
                  f"'{template_id}' (calibration)")
        elif alignment_cal.get('initial_transform'):
            initial_transform_cal = alignment_cal['initial_transform']
            template_id = args.use_initial_transform
            print(f"Loaded initial transform template from sample "
                  f"'{template_id}' (calibration)")

        if initial_transform_main is None and initial_transform_cal is None:
            template_id = args.use_initial_transform
            print(f"Warning: No alignment found in template sample "
                  f"'{template_id}'")
        print()

    # Find samples
    if args.all:
        # Find all samples with config.json
        samples = sorted(
            [
                d.name
                for d in samples_dir.iterdir()
                if d.is_dir() and (d / "config.json").exists()
            ]
        )
    else:
        samples = args.samples

    # Filter to only rejected samples if --rejected-only is set
    if args.rejected_only:
        rejected_samples = []
        for sample_id in samples:
            config = load_sample_config(sample_id, args.samples_dir)
            if config is None:
                continue

            # Check if main alignment is rejected
            alignment_main = config.get("alignment_main", {})
            main_rejected = alignment_main.get("review_status") == "rejected"

            # Check if calibration alignment is rejected
            alignment_cal = config.get("alignment_calibration", {})
            cal_rejected = alignment_cal.get("review_status") == "rejected"

            if main_rejected or cal_rejected:
                rejected_samples.append(sample_id)

        samples = rejected_samples

        if not samples:
            print("No rejected samples found.")
            print("Run step1c to review alignments first:")
            print("  python scripts/step1c_review_alignments.py --all")
            sys.exit(0)

    if not samples:
        print("Error: No samples to process")
        sys.exit(1)

    print("=" * 70)
    print("Step 1b: Run Automatic Alignments (Automated)")
    print("=" * 70)
    print(f"Samples: {len(samples)}")
    if args.rejected_only:
        print("Mode: Reprocessing REJECTED samples only")
    elif args.preview_only:
        print("Mode: Generating previews only "
              "(no alignment calculation)")
    print(f"Main video method: {args.main_method}")
    print(f"Main motion type: {args.main_motion_type}")
    temporal_status = 'disabled' if args.no_temporal else 'enabled'
    print(f"Temporal alignment: {temporal_status}")
    print(f"Calibration video method: {args.calibration_method}")
    print(f"Calibration motion type: {args.calibration_motion_type}")
    print()

    results = []

    for i, sample_id in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] Sample {sample_id}")
        print("-" * 70)

        # Load sample config
        config = load_sample_config(sample_id, args.samples_dir)
        if config is None:
            print("  ⚠ No config.json found, skipping")
            cmd = "python scripts/step1a_select_flips.py"
            print(f"    Run: {cmd} --samples {sample_id}")
            results.append((sample_id, "SKIPPED", "No config"))
            continue

        flip_main = config.get("flip_main")
        if flip_main is None:
            print("  ⚠ No flip configured, skipping")
            results.append((sample_id, "SKIPPED", "No flip"))
            continue

        print(f"  Flip (main): {flip_main}")

        sample_dir = Path(samples_dir) / sample_id

        # Process main videos (skip if --calibration-only)
        if not args.calibration_only:
            print("\n  Main videos:")
            vis_path, uv_path = find_video_pair(
                sample_dir, use_calibration=False
            )

            if vis_path is None:
                print("    ✗ Not found")
                if not config.get("has_calibration"):
                    # No main videos and no calibration videos - failure
                    failure_msg = "Main videos not found"
                    results.append((sample_id, "FAILED", failure_msg))
                    continue
                else:
                    # Has calibration videos, we'll process those instead
                    print("    Skipping main (--calibration-only)")
            else:
                print(f"    VIS: {Path(vis_path).name}")
                print(f"    UV:  {Path(uv_path).name}")

            # Check if this specific alignment was rejected
            alignment_main_info = config.get("alignment_main", {})
            main_rejected = (
                args.rejected_only and
                alignment_main_info.get("review_status") == "rejected"
            )

            # Check if already done or preview-only mode
            if args.preview_only:
                # Preview-only: generate preview from existing alignment
                alignment_main = config.get("alignment_main")
                if not alignment_main:
                    msg = "run without --preview-only first"
                    print(f"    ✗ No alignment found ({msg})")
                    results.append((sample_id, "FAILED",
                                    "No alignment for preview"))
                    continue

                print("    Generating preview from existing alignment...")
                try:
                    generate_preview_from_alignment(
                        vis_path,
                        uv_path,
                        flip_main,
                        alignment_main.get("homography_matrix"),
                        alignment_main.get("output_size"),
                        sample_dir,
                        sample_id,
                        video_type="main",
                    )
                except Exception as e:
                    print(f"    ✗ Failed to generate preview: {e}")
                    error_msg = f"Preview failed: {str(e)[:30]}"
                    results.append((sample_id, "FAILED", error_msg))
                    continue

            elif (config.get("alignment_main") and
                  not args.force and
                  not main_rejected):
                print("    ✓ Already aligned (use --force to reprocess)")
            else:
                # Show why we're reprocessing
                if main_rejected:
                    print("    ⚠ Alignment was rejected - reprocessing...")
                elif args.force:
                    pass  # Already indicated by the command

                # Check if this sample's config has initial_transform
                sample_initial_transform = None
                alignment_main_cfg = config.get("alignment_main", {})
                if alignment_main_cfg.get("initial_transform"):
                    sample_initial_transform = (
                        alignment_main_cfg["initial_transform"]
                    )
                    print("    Using initial_transform from config.json")

                # Use sample's initial_transform if available, otherwise
                # use the global one from --use-initial-transform
                transform_to_use = (sample_initial_transform if
                                    sample_initial_transform else
                                    initial_transform_main)

                print("    Calculating alignment... ", end="", flush=True)
                start = time.time()

                try:
                    fitted_op = run_alignment(
                        vis_path,
                        uv_path,
                        flip_main,
                        method=args.main_method,
                        temporal=not args.no_temporal,
                        save_preview=args.save_preview,
                        preview_dir=sample_dir,
                        sample_id=sample_id,
                        video_type="main",
                        motion_type=args.main_motion_type,
                        initial_transform=transform_to_use,
                    )
                    elapsed = time.time() - start
                    print(f"✓ Done in {elapsed:.1f}s")

                    if args.save_preview:
                        print("    Previews saved:")
                        print(f"      - {sample_id}_main_vis.png")
                        print(f"      - {sample_id}_main_uv_aligned.png")
                        print(f"      - {sample_id}_main_composite.png")

                    # Save alignment parameters to config for reuse
                    has_coe = (hasattr(fitted_op, "coe") and
                               fitted_op.coe is not None)
                    alignment_params = {
                        "method": args.main_method,
                        "motion_type": args.main_motion_type,
                        "flip": flip_main,
                        "has_homography": has_coe,
                        "temporal_shift": (
                            fitted_op.time_shift
                            if hasattr(fitted_op, "time_shift")
                            else None
                        ),
                        "processing_time": elapsed,
                        "vis_frame_count": fitted_op.vis_frame_count,
                        "uv_frame_count": fitted_op.uv_frame_count,
                        "usable_frame_count": fitted_op.usable_frame_count,
                    }

                    # Save the actual homography matrix for reuse
                    if hasattr(fitted_op, "coe") and fitted_op.coe is not None:
                        alignment_params["homography_matrix"] = (
                            fitted_op.coe.tolist()
                        )

                    # Save output size
                    if hasattr(fitted_op, "output_size"):
                        alignment_params["output_size"] = (
                            fitted_op.output_size
                        )

                    # Optionally save as template for future use
                    save_template = (args.save_as_template and
                                     hasattr(fitted_op, "coe") and
                                     fitted_op.coe is not None)
                    if save_template:
                        alignment_params["initial_transform"] = (
                            fitted_op.coe.tolist()
                        )

                    config["alignment_main"] = alignment_params

                    # Show alignment params
                    print(f"    Motion type: {args.main_motion_type}")
                    if transform_to_use is not None:
                        print("    Used initial transform as starting point")
                    if alignment_params["has_homography"]:
                        print("    Transform matrix saved (3x3)")
                    if args.save_as_template:
                        msg = "can be used with --use-initial-transform"
                        print(f"    Saved as template ({msg})")
                    vis_count = alignment_params["vis_frame_count"]
                    uv_count = alignment_params["uv_frame_count"]
                    if vis_count != uv_count:
                        print("    ⚠ Frame count mismatch detected:")
                        print(f"      VIS: {vis_count} frames")
                        print(f"      UV:  {uv_count} frames")
                        usable = alignment_params['usable_frame_count']
                        print(f"      Usable: {usable} frames")
                    if alignment_params["temporal_shift"] is not None:
                        if alignment_params["temporal_shift"] == 0:
                            msg = "no temporal alignment"
                            print(f"    Temporal shift: 0 frames ({msg})")
                        else:
                            shift = alignment_params['temporal_shift']
                            print(f"    Temporal shift: {shift} frames")

                except AlignmentNotFound as e:
                    print(f"✗ Failed: {e}")
                    error_msg = f"Main: {str(e)[:50]}"
                    results.append((sample_id, "FAILED", error_msg))
                    continue
                except Exception as e:
                    import traceback

                    print(f"✗ Error: {e}")
                    traceback.print_exc()
                    error_msg = f"Main: {str(e)[:50]}"
                    results.append((sample_id, "ERROR", error_msg))
                    continue

        # Process calibration videos
        if not args.skip_calibration and config.get("has_calibration"):
            flip_cal = config.get("flip_calibration")

            # Check if calibration alignment was rejected
            alignment_cal_info = config.get("alignment_calibration", {})
            cal_rejected = (
                args.rejected_only and
                alignment_cal_info.get("review_status") == "rejected"
            )

            if flip_cal is None:
                msg = "Calibration videos exist but no flip configured"
                print(f"\n  ⚠ {msg}")
            elif args.preview_only:
                # Preview-only mode for calibration
                alignment_cal = config.get("alignment_calibration")
                if not alignment_cal:
                    print("\n  Calibration videos:")
                    msg = "run without --preview-only first"
                    print(f"    ✗ No alignment found ({msg})")
                else:
                    print("\n  Calibration videos:")
                    print(f"    Flip: {flip_cal}")

                    pair = find_video_pair(
                        sample_dir, use_calibration=True
                    )
                    vis_cal_path, uv_cal_path = pair

                    if vis_cal_path is None:
                        print("    ⚠ Not found, skipping")
                    else:
                        print(f"    VIS: {Path(vis_cal_path).name}")
                        print(f"    UV:  {Path(uv_cal_path).name}")
                        msg = "Generating preview from existing alignment..."
                        print(f"    {msg}")

                        try:
                            generate_preview_from_alignment(
                                vis_cal_path,
                                uv_cal_path,
                                flip_cal,
                                alignment_cal.get("homography_matrix"),
                                alignment_cal.get("output_size"),
                                sample_dir,
                                sample_id,
                                video_type="calibration",
                            )
                        except Exception as e:
                            print(f"    ✗ Failed to generate preview: {e}")

            elif (
                config.get("alignment_calibration")
                and not args.force
                and not cal_rejected
            ):
                print("\n  Calibration videos:")
                print("    ✓ Already aligned (use --force to reprocess)")
            else:
                print("\n  Calibration videos:")
                # Show why we're reprocessing
                if cal_rejected:
                    print("    ⚠ Alignment was rejected - reprocessing...")
                print(f"    Flip: {flip_cal}")

                pair = find_video_pair(
                    sample_dir, use_calibration=True
                )
                vis_cal_path, uv_cal_path = pair

                if vis_cal_path is None:
                    print("    ⚠ Not found, skipping")
                else:
                    print(f"    VIS: {Path(vis_cal_path).name}")
                    print(f"    UV:  {Path(uv_cal_path).name}")

                    # Check if this sample's config has initial_transform
                    # for calibration
                    sample_cal_initial_transform = None
                    alignment_cal_cfg = config.get(
                        "alignment_calibration", {}
                    )
                    if alignment_cal_cfg.get("initial_transform"):
                        sample_cal_initial_transform = (
                            alignment_cal_cfg["initial_transform"]
                        )
                        print("    Using initial_transform from config.json")

                    # Use sample's initial_transform if available, otherwise
                    # use the global one from --use-initial-transform
                    cal_transform_to_use = (sample_cal_initial_transform if
                                            sample_cal_initial_transform else
                                            initial_transform_cal)

                    print("    Calculating alignment... ", end="", flush=True)

                    start = time.time()
                    try:
                        # Calibration videos don't need temporal
                        fitted_cal_op = run_alignment(
                            vis_cal_path,
                            uv_cal_path,
                            flip_cal,
                            method=args.calibration_method,
                            temporal=False,
                            save_preview=args.save_preview,
                            preview_dir=sample_dir,
                            sample_id=sample_id,
                            video_type="calibration",
                            motion_type=args.calibration_motion_type,
                            initial_transform=cal_transform_to_use,
                        )
                        elapsed = time.time() - start
                        print(f"✓ Done in {elapsed:.1f}s")

                        if args.save_preview:
                            print("    Previews saved:")
                            vis_file = f"{sample_id}_calibration_vis.png"
                            uv_name = "_calibration_uv_aligned.png"
                            uv_file = f"{sample_id}{uv_name}"
                            comp_name = "_calibration_composite.png"
                            comp_file = f"{sample_id}{comp_name}"
                            print(f"      - {vis_file}")
                            print(f"      - {uv_file}")
                            print(f"      - {comp_file}")

                        # Save calibration alignment parameters
                        has_coe_cal = (hasattr(fitted_cal_op, "coe") and
                                       fitted_cal_op.coe is not None)
                        cal_alignment_params = {
                            "method": args.calibration_method,
                            "motion_type": args.calibration_motion_type,
                            "flip": flip_cal,
                            "has_homography": has_coe_cal,
                            "temporal_shift": (
                                fitted_cal_op.time_shift
                                if hasattr(fitted_cal_op, "time_shift")
                                else 0
                            ),
                            "processing_time": elapsed,
                        }

                        # Save the actual homography matrix
                        if has_coe_cal:
                            cal_alignment_params["homography_matrix"] = (
                                fitted_cal_op.coe.tolist()
                            )

                        # Save output size
                        if hasattr(fitted_cal_op, "output_size"):
                            cal_alignment_params["output_size"] = (
                                fitted_cal_op.output_size
                            )

                        # Optionally save as template for future use
                        save_template_cal = (args.save_as_template and
                                             has_coe_cal)
                        if save_template_cal:
                            cal_alignment_params["initial_transform"] = (
                                fitted_cal_op.coe.tolist()
                            )

                        config["alignment_calibration"] = cal_alignment_params

                        motion = args.calibration_motion_type
                        print(f"    Motion type: {motion}")
                        if cal_transform_to_use is not None:
                            msg = "Used initial transform as starting point"
                            print(f"    {msg}")
                        if args.save_as_template:
                            msg = "can be used with --use-initial-transform"
                            print(f"    Saved as template ({msg})")
                        if cal_alignment_params["has_homography"]:
                            print("    Transform matrix saved (3x3)")
                        if cal_alignment_params["temporal_shift"] != 0:
                            shift = cal_alignment_params['temporal_shift']
                            print(f"    Temporal shift: {shift} frames")

                    except Exception as e:
                        print(f"⚠ Failed: {e}")
                        config["alignment_calibration"] = {"error": str(e)}

        # Save updated config
        config_path = sample_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        results.append((sample_id, "SUCCESS", "Aligned"))

    # Summary
    sep = "=" * 70
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)

    success_count = sum(1 for _, status, _ in results
                        if status == "SUCCESS")
    failed_statuses = ["FAILED", "ERROR"]
    failed_count = sum(1 for _, status, _ in results
                       if status in failed_statuses)
    skipped_count = sum(1 for _, status, _ in results
                        if status == "SKIPPED")

    for sample_id, status, detail in results:
        symbol = (
            "✓"
            if status == "SUCCESS"
            else "✗" if status in ["FAILED", "ERROR"] else "⚠"
        )
        print(f"{symbol} {sample_id}: {status} - {detail}")

    print(f"\nSuccess: {success_count}/{len(results)}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")

    if success_count > 0:
        if args.preview_only:
            print("\n✓ Preview images generated from existing alignments")
        else:
            print("\n✓ Alignment parameters saved to config.json files")
            if args.save_preview:
                print("✓ Preview images saved to sample directories")
        print("\nNext step:")
        print("  python scripts/step1c_review_alignments.py --all")


if __name__ == "__main__":
    main()
