#!/usr/bin/env python3
"""
Quick setup checker for batch video processing workflow.

Verifies that all required files and directories are in place.

Usage (from project root):
    python -m scripts.check_setup

    Or if video2vision is installed:
    python scripts/check_setup.py
"""

import sys
from pathlib import Path


def check_mark(passed):
    return "✓" if passed else "✗"


def main():
    print("=" * 70)
    print("Video2Vision Batch Workflow Setup Check")
    print("=" * 70)

    issues = []

    # Check package installation
    print("\n1. Package Installation")
    try:
        import video2vision  # noqa: F401

        print(f"   {check_mark(True)} video2vision package installed")
    except ImportError:
        print(f"   {check_mark(False)} video2vision package NOT installed")
        issues.append("Run: python3 -m pip install .")

    try:
        import cv2  # noqa: F401

        print(f"   {check_mark(True)} opencv installed")
    except ImportError:
        print(f"   {check_mark(False)} opencv NOT installed")
        issues.append(
            "Run: python3 -m pip install opencv-contrib-python-headless"
        )

    try:
        import numpy  # noqa: F401

        print(f"   {check_mark(True)} numpy installed")
    except ImportError:
        print(f"   {check_mark(False)} numpy NOT installed")
        issues.append("Run: python3 -m pip install numpy")

    # Check directory structure
    print("\n2. Directory Structure")

    video_samples = Path("videos/samples")
    if video_samples.exists():
        samples = sorted(
            [d.name for d in video_samples.iterdir() if d.is_dir()]
        )
        print(f"   {check_mark(True)} videos/samples/ exists")
        samples_str = ', '.join(samples[:5])
        print(f"      Found {len(samples)} sample(s): {samples_str}")
        if len(samples) > 5:
            print(f"      ... and {len(samples) - 5} more")
    else:
        print(f"   {check_mark(False)} videos/samples/ NOT found")
        issues.append("Create videos/samples/ and organize your videos")

    data_dir = Path("data")
    if data_dir.exists():
        print(f"   {check_mark(True)} data/ directory exists")

        # Check for required data files
        camera_sens = data_dir / "camera_sensitivities.csv"
        if camera_sens.exists():
            print(f"   {check_mark(True)} camera_sensitivities.csv found")
        else:
            print(f"   {check_mark(False)} camera_sensitivities.csv NOT found")
            issues.append("Need camera sensitivity data")

        aruco_samples = data_dir / "aruco_samples.csv"
        if aruco_samples.exists():
            print(f"   {check_mark(True)} aruco_samples.csv found")
        else:
            print(f"   {check_mark(False)} aruco_samples.csv NOT found")
            issues.append(
                "Need ArUco sample values (or your calibration target data)"
            )
    else:
        print(f"   {check_mark(False)} data/ directory NOT found")
        issues.append("Need data directory with calibration files")

    # Check for animal sensitivities
    animal_sens_dir = Path("data/animal_sensitivities")
    if animal_sens_dir.exists():
        animals = sorted(
            [
                f.stem.replace("_sensitivities", "")
                for f in animal_sens_dir.glob("*_sensitivities.csv")
            ]
        )
        print(f"   {check_mark(True)} Animal sensitivities found")
        print(f"      Available: {', '.join(animals[:3])}")
        if len(animals) > 3:
            print(f"      ... and {len(animals) - 3} more")
    else:
        print(f"   {check_mark(False)} Animal sensitivities NOT found")

    # Check video pairs
    print("\n3. Video Pairs")
    if video_samples.exists() and samples:
        sample_to_check = samples[0]
        sample_dir = video_samples / sample_to_check

        vis_videos = list(sample_dir.glob("VIS_*.MP4")) + list(
            sample_dir.glob("VIS_*.mp4")
        )
        uv_videos = list(sample_dir.glob("UV_*.MP4")) + list(
            sample_dir.glob("UV_*.mp4")
        )

        if vis_videos and uv_videos:
            msg = f"Sample {sample_to_check} has video pairs"
            print(f"   {check_mark(True)} {msg}")
            print(f"      VIS: {vis_videos[0].name}")
            print(f"      UV:  {uv_videos[0].name}")
        else:
            msg = f"Sample {sample_to_check} missing videos"
            print(f"   {check_mark(False)} {msg}")
            issues.append("Videos should be named VIS_*.MP4 and UV_*.MP4")
    else:
        print(f"   {check_mark(False)} No samples to check")

    # Summary
    print("\n" + "=" * 70)
    if issues:
        print("ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\nFix these issues before running the workflow.")
        return 1
    else:
        print("✓ ALL CHECKS PASSED")
        print("\nYou're ready to start processing!")
        print("\nNext step:")
        print("  python3 scripts/step1_create_alignment.py --sample 001")
        print("\nOr see full workflow:")
        print("  cat scripts/BATCH_WORKFLOW.md")
        return 0


if __name__ == "__main__":
    sys.exit(main())
