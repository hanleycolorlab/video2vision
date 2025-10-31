#!/usr/bin/env python3
"""
Step 1a: Select flip directions interactively with OpenCV windows.

Shows side-by-side VIS/UV comparison in a window.
Press 'n' to cycle through flip options, SPACE/ENTER to confirm.

Usage (from project root):
    python -m scripts.step1a_select_flips --samples 001
    python -m scripts.step1a_select_flips --all

    Or if video2vision is installed:
    python scripts/step1a_select_flips.py --samples 001
    python scripts/step1a_select_flips.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_first_frame(video_path):
    """Load first frame from video"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read video: {video_path}")

    # Convert BGR to RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def create_2x2_grid(vis_img, uv_img, sample_id):
    """Create 2x2 grid: VIS in top-left, UV with 3 flip options"""
    # Apply flips
    uv_none = uv_img.copy()
    uv_h = cv2.flip(uv_img, 1)
    uv_v = cv2.flip(uv_img, 0)

    images = [
        (vis_img, "VISIBLE", (255, 255, 255)),
        (uv_none, "UV - none", (0, 255, 0)),
        (uv_h, "UV - horizontal", (0, 255, 0)),
        (uv_v, "UV - vertical", (0, 255, 0)),
    ]

    # Process each image
    displays = []
    for img, label, color in images:
        # Ensure uint8
        if img.dtype != np.uint8:
            disp = (np.clip(img, 0, 255)).astype(np.uint8)
        else:
            disp = img.copy()

        # Convert RGB to BGR
        disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)

        # Resize
        max_h = 600
        if disp.shape[0] > max_h:
            scale = max_h / disp.shape[0]
            disp = cv2.resize(disp, None, fx=scale, fy=scale)

        # Add label
        label_h = 40
        label_img = np.zeros((label_h, disp.shape[1], 3), dtype=np.uint8)
        cv2.putText(label_img, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        disp = np.vstack([label_img, disp])

        displays.append(disp)

    # Arrange in 2x2 grid
    top_row = np.hstack(displays[:2])
    bottom_row = np.hstack(displays[2:])

    # Make rows same width
    if top_row.shape[1] < bottom_row.shape[1]:
        pad_width = bottom_row.shape[1] - top_row.shape[1]
        pad = np.zeros((top_row.shape[0], pad_width, 3), dtype=np.uint8)
        top_row = np.hstack([top_row, pad])
    elif bottom_row.shape[1] < top_row.shape[1]:
        pad_width = top_row.shape[1] - bottom_row.shape[1]
        pad = np.zeros((bottom_row.shape[0], pad_width, 3), dtype=np.uint8)
        bottom_row = np.hstack([bottom_row, pad])

    combined = np.vstack([top_row, bottom_row])

    # Add instructions at bottom
    instruction_h = 60
    instructions = np.zeros(
        (instruction_h, combined.shape[1], 3), dtype=np.uint8
    )
    text = f"Sample {sample_id}: Click the UV that matches VISIBLE orientation"
    cv2.putText(instructions, text,
                (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(instructions, "[q] Quit | [s] Skip",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    combined = np.vstack([combined, instructions])

    # Calculate click regions (which quadrant was clicked)
    # Returns region boundaries
    mid_y = (top_row.shape[0])
    mid_x = combined.shape[1] // 2

    return combined, mid_x, mid_y


def select_flip_for_sample(sample_id, sample_dir, video_type="main"):
    """Interactive flip selection for one sample - click to select

    Args:
        sample_id: Sample identifier
        sample_dir: Path to sample directory
        video_type: "main" or "calibration"
    """
    sample_dir = Path(sample_dir)

    # Find videos
    if video_type == "calibration":
        search_dir = sample_dir / "calibration"
        label_prefix = f"{sample_id} CALIBRATION"
    else:
        search_dir = sample_dir
        label_prefix = sample_id

    vis_videos = (sorted(search_dir.glob('VIS_*.MP4')) +
                  sorted(search_dir.glob('VIS_*.mp4')))
    uv_videos = (sorted(search_dir.glob('UV_*.MP4')) +
                 sorted(search_dir.glob('UV_*.mp4')))

    if not vis_videos or not uv_videos:
        print(f"⚠ {label_prefix}: No video pair found")
        return None

    vis_path = str(vis_videos[0])
    uv_path = str(uv_videos[0])

    print(f"\n{label_prefix}: Loading...")

    # Load first frames
    try:
        vis_img = load_first_frame(vis_path)
        uv_img = load_first_frame(uv_path)
    except Exception as e:
        print(f"✗ Error loading videos: {e}")
        return None

    # Create 2x2 grid
    combined, mid_x, mid_y = create_2x2_grid(vis_img, uv_img, label_prefix)

    window_name = f"{label_prefix}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 1000)

    selected_flip = [None]  # Use list to modify in callback

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Determine which quadrant was clicked
            # Top-left: VISIBLE (ignore click)
            # Top-right: UV none
            # Bottom-left: UV horizontal
            # Bottom-right: UV vertical

            if y < mid_y:  # Top row
                if x >= mid_x:  # Top-right
                    selected_flip[0] = 'none'
            else:  # Bottom row
                if x < mid_x:  # Bottom-left
                    selected_flip[0] = 'horizontal'
                else:  # Bottom-right
                    selected_flip[0] = 'vertical'

    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, combined)

    print("  Click on the UV image that matches VISIBLE | [q] Quit | [s] Skip")

    while True:
        key = cv2.waitKey(50) & 0xFF

        # Check if selection made
        if selected_flip[0] is not None:
            cv2.destroyWindow(window_name)
            print(f"  ✓ Selected: {selected_flip[0]}")
            return selected_flip[0]

        # Handle keyboard
        if key == ord('s'):  # Skip
            cv2.destroyWindow(window_name)
            print("  ⊘ Skipped")
            return None
        elif key == ord('q') or key == 27:  # Quit
            cv2.destroyAllWindows()
            print("\n✗ Quit by user")
            return "QUIT"


def main():
    parser = argparse.ArgumentParser(
        description=("Select flip directions interactively - "
                     "saves config.json in each sample folder")
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--all', action='store_true', help='Process all samples'
    )
    group.add_argument(
        '--samples', nargs='+', help='Specific samples (e.g., 001 006 012)'
    )

    parser.add_argument(
        '--samples-dir', default='videos/samples',
        help='Directory containing samples (default: videos/samples)'
    )

    args = parser.parse_args()

    # Find samples
    if args.all:
        sample_dir = Path(args.samples_dir)
        if not sample_dir.exists():
            print(f"Error: {sample_dir} not found")
            sys.exit(1)
        samples = sorted([d.name for d in sample_dir.iterdir() if d.is_dir()])
    else:
        samples = args.samples

    if not samples:
        print("Error: No samples found")
        sys.exit(1)

    print("="*70)
    print("Step 1a: Select Flip Directions")
    print("="*70)
    print(f"Samples: {len(samples)}")
    print("\nFor each sample, a 2x2 grid will show:")
    print("  Top-left:     VISIBLE image (reference)")
    print("  Top-right:    UV - none")
    print("  Bottom-left:  UV - horizontal flip")
    print("  Bottom-right: UV - vertical flip")
    print("\nClick on the UV image that matches VISIBLE orientation")
    print("Keyboard: [s] Skip | [q] Quit")
    print(f"\nConfig saved to: {args.samples_dir}/{{sample_id}}/config.json")
    print()

    results = {}

    for i, sample_id in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] Sample {sample_id}")
        print("=" * 70)

        sample_dir = Path(args.samples_dir) / sample_id

        if not sample_dir.exists():
            print("⚠ Directory not found, skipping")
            continue

        # Check if has calibration
        has_calibration = (
            (sample_dir / 'calibration').exists()
        )

        # Main videos
        print("\nMAIN VIDEOS")
        print("-" * 70)
        main_flip = select_flip_for_sample(
            sample_id, sample_dir, video_type="main"
        )

        if main_flip == "QUIT":
            break
        elif main_flip is None:
            # Skipped main, skip calibration too
            continue

        # Calibration videos (if they exist)
        cal_flip = None
        if has_calibration:
            print("\nCALIBRATION VIDEOS")
            print("-" * 70)
            cal_flip = select_flip_for_sample(
                sample_id, sample_dir, video_type="calibration"
            )

            if cal_flip == "QUIT":
                break

        # Save config
        if main_flip is not None:
            # Load existing config if it exists
            config_path = sample_dir / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {
                    'sample_id': sample_id,
                    'created': str(
                        Path(sample_dir).stat().st_mtime
                    ),
                }

            # Update with flip info for main videos
            config['flip_main'] = main_flip
            config['has_calibration'] = has_calibration

            # Find main video files
            vis_videos = (sorted(sample_dir.glob('VIS_*.MP4')) +
                          sorted(sample_dir.glob('VIS_*.mp4')))
            uv_videos = (sorted(sample_dir.glob('UV_*.MP4')) +
                         sorted(sample_dir.glob('UV_*.mp4')))

            config['main_videos'] = {
                'vis': vis_videos[0].name if vis_videos else None,
                'uv': uv_videos[0].name if uv_videos else None,
            }

            # Add calibration info if present
            if has_calibration and cal_flip is not None:
                config['flip_calibration'] = cal_flip

                cal_dir = sample_dir / 'calibration'
                cal_vis = (sorted(cal_dir.glob('VIS_*.MP4')) +
                           sorted(cal_dir.glob('VIS_*.mp4')))
                cal_uv = (sorted(cal_dir.glob('UV_*.MP4')) +
                          sorted(cal_dir.glob('UV_*.mp4')))

                config['calibration_videos'] = {
                    'vis': cal_vis[0].name if cal_vis else None,
                    'uv': cal_uv[0].name if cal_uv else None,
                }

            # Save to sample folder
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"  ✓ Saved: {config_path}")

            results[sample_id] = config

    # Summary
    print("\n" + "=" * 70)
    if results:
        print("SUMMARY")
        print("=" * 70)
        print(f"Processed: {len(results)}/{len(samples)} samples")
        print()

        flip_counts = {}
        for sample_id, config in results.items():
            main_flip = config.get('flip_main', 'unknown')
            cal_flip = config.get('flip_calibration', 'N/A')
            flip_counts[main_flip] = (
                flip_counts.get(main_flip, 0) + 1
            )
            print(
                f"  {sample_id}: main={main_flip}, cal={cal_flip}"
            )

        print("\nFlip distribution:")
        for flip, count in sorted(flip_counts.items()):
            print(f"  {flip}: {count} samples")

        print(
            f"\n✓ Configs saved to: {args.samples_dir}/{{sample_id}}"
            "/config.json"
        )
        print("\nNext steps:")
        print("  # Preview alignment:")
        samples_list = ' '.join(list(results.keys())[:3])
        print(
            f"  python scripts/step1c_preview_alignment.py "
            f"--samples {samples_list}"
        )
        print("\n  # Run full batch alignment:")
        print(
            "  python scripts/step1b_run_alignments.py --all --method aruco"
        )
    else:
        print("SUMMARY")
        print("=" * 70)
        print("No samples processed")


if __name__ == '__main__':
    main()
