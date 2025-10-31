#!/usr/bin/env python3
"""
Step 1c: Review alignment composite images (MANUAL - QUICK).

This displays the composite images created by step1b so you can visually
verify the alignment quality. Use keyboard to approve/reject each sample.

Usage (from project root):
    # Using module syntax:
    python -m scripts.step1c_review_alignments --all
    python -m scripts.step1c_review_alignments --samples 001 006 012

    # Or if video2vision is installed:
    python scripts/step1c_review_alignments.py --all
    python scripts/step1c_review_alignments.py --samples 001 006 012

    # Skip main/calibration
    python scripts/step1c_review_alignments.py --all --skip-calibration
    python scripts/step1c_review_alignments.py --all --main-only

Controls:
    Y or SPACE = Approve this alignment
    N or R     = Reject this alignment
    S          = Skip (leave unmarked)
    Q or ESC   = Quit review
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_sample_config(sample_id, samples_dir='videos/samples'):
    """Load configuration for a sample"""
    config_path = Path(samples_dir) / sample_id / 'config.json'

    if not config_path.exists():
        return None

    with open(config_path, 'r') as f:
        return json.load(f)


def save_sample_config(sample_id, config, samples_dir='videos/samples'):
    """Save configuration for a sample"""
    config_path = Path(samples_dir) / sample_id / 'config.json'

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def find_composite_images(sample_id, video_type, samples_dir='videos/samples'):
    """Find the composite image for a sample"""
    sample_dir = Path(samples_dir) / sample_id

    composite_path = sample_dir / f"{sample_id}_{video_type}_composite.png"

    if composite_path.exists():
        return composite_path

    return None


def review_composite(sample_id, video_type, composite_path):
    """Show composite image and get user approval"""
    # Load composite
    composite = cv2.imread(str(composite_path))

    if composite is None:
        print(f"    ✗ Could not load composite image")
        return None

    # Create instruction overlay
    h, w = composite.shape[:2]
    overlay = composite.copy()

    # Add dark bar at top for instructions
    bar_height = 60
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)

    # Instructions text
    text1 = f"Sample {sample_id} - {video_type.upper()}"
    text2 = "[Y/SPACE]=Approve  [N/R]=Reject  [S]=Skip  [Q/ESC]=Quit"

    cv2.putText(overlay, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Show window
    window_name = f"Review: {sample_id} - {video_type}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, overlay)

    # Wait for key
    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('y') or key == ord('Y') or key == ord(' '):  # Approve
            cv2.destroyWindow(window_name)
            return 'approved'
        elif key == ord('n') or key == ord('N') or key == ord('r') or key == ord('R'):  # Reject
            cv2.destroyWindow(window_name)
            return 'rejected'
        elif key == ord('s') or key == ord('S'):  # Skip
            cv2.destroyWindow(window_name)
            return 'skipped'
        elif key == ord('q') or key == ord('Q') or key == 27:  # Quit
            cv2.destroyWindow(window_name)
            return 'quit'


def main():
    parser = argparse.ArgumentParser(
        description="Review alignment composite images and approve/reject"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Review all samples with composites')
    group.add_argument('--samples', nargs='+', help='Specific samples to review')

    parser.add_argument('--skip-calibration', action='store_true',
                       help='Only review main videos')
    parser.add_argument('--main-only', action='store_true',
                       help='Only review main videos (same as --skip-calibration)')
    parser.add_argument('--samples-dir', default='videos/samples',
                       help='Directory containing samples (default: videos/samples)')

    args = parser.parse_args()

    # Find samples
    if args.all:
        samples_dir = Path(args.samples_dir)
        if not samples_dir.exists():
            print(f"Error: Samples directory not found: {samples_dir}")
            sys.exit(1)

        # Find all samples with composites
        samples = sorted([
            d.name for d in samples_dir.iterdir()
            if d.is_dir() and list(d.glob('*_composite.png'))
        ])
    else:
        samples = args.samples

    if not samples:
        print("Error: No samples to review")
        sys.exit(1)

    print("="*70)
    print("Step 1c: Review Alignment Composites")
    print("="*70)
    print(f"Samples: {len(samples)}")
    print()
    print("Check the composite images to verify alignment quality:")
    print("  - UV and VIS should overlap perfectly")
    print("  - No ghosting or color fringing")
    print("  - Features should line up (edges, landmarks, etc.)")
    print()

    skip_calibration = args.skip_calibration or args.main_only

    approved_count = 0
    rejected_count = 0
    skipped_count = 0

    for i, sample_id in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] Sample {sample_id}")
        print("-" * 70)

        # Load config
        config = load_sample_config(sample_id, args.samples_dir)
        if config is None:
            print(f"  ⚠ No config.json found, skipping")
            continue

        # Check if alignment exists
        if not config.get('alignment_main'):
            print(f"  ⚠ No alignment calculated yet")
            print(f"    Run: python scripts/step1b_run_alignments.py --samples {sample_id} --save-preview")
            continue

        # Review main videos
        print(f"\n  Main videos:")
        main_composite = find_composite_images(sample_id, 'main', args.samples_dir)

        if main_composite is None:
            print(f"    ⚠ No composite image found")
            print(f"    Run: python scripts/step1b_run_alignments.py --samples {sample_id} --save-preview --force")
        else:
            print(f"    Showing: {main_composite.name}")
            result = review_composite(sample_id, 'main', main_composite)

            if result == 'quit':
                print("\n✗ Review quit by user")
                break
            elif result == 'approved':
                print(f"    ✓ Approved")
                config['alignment_main']['review_status'] = 'approved'
                approved_count += 1
            elif result == 'rejected':
                print(f"    ✗ Rejected")
                config['alignment_main']['review_status'] = 'rejected'
                rejected_count += 1
            elif result == 'skipped':
                print(f"    ⊘ Skipped")
                skipped_count += 1

        # Review calibration videos if present
        if not skip_calibration and config.get('has_calibration') and config.get('alignment_calibration'):
            print(f"\n  Calibration videos:")
            cal_composite = find_composite_images(sample_id, 'calibration', args.samples_dir)

            if cal_composite is None:
                print(f"    ⚠ No composite image found")
            else:
                print(f"    Showing: {cal_composite.name}")
                result = review_composite(sample_id, 'calibration', cal_composite)

                if result == 'quit':
                    print("\n✗ Review quit by user")
                    # Save config before breaking
                    save_sample_config(sample_id, config, args.samples_dir)
                    break
                elif result == 'approved':
                    print(f"    ✓ Approved")
                    config['alignment_calibration']['review_status'] = 'approved'
                    approved_count += 1
                elif result == 'rejected':
                    print(f"    ✗ Rejected")
                    config['alignment_calibration']['review_status'] = 'rejected'
                    rejected_count += 1
                elif result == 'skipped':
                    print(f"    ⊘ Skipped")
                    skipped_count += 1

        # Save updated config
        save_sample_config(sample_id, config, args.samples_dir)

    # Summary
    print("\n" + "="*70)
    print("REVIEW SUMMARY")
    print("="*70)
    print(f"Approved: {approved_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Skipped:  {skipped_count}")
    print()

    if approved_count > 0:
        print(f"✓ {approved_count} alignments approved!")
        print()
        print("Next step: Apply alignments to full videos (overnight)")
        print("  python scripts/step2_apply_alignments.py --approved-only")

    if rejected_count > 0:
        print()
        print(f"⚠ {rejected_count} alignments rejected")
        print("  Reprocess rejected samples with different settings:")
        print("  python scripts/step1b_run_alignments.py --all --rejected-only --save-preview --main-motion-type euclidean")
        print()
        print("  Or for specific samples:")
        print("  python scripts/step1b_run_alignments.py --samples <ID> --save-preview --main-motion-type euclidean --force")


if __name__ == '__main__':
    main()
