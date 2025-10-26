#!/usr/bin/env python3
"""
Create a custom autolinearizer from a calibration frame.

This script lets you manually position the 28 (or 24) color patches on a reference
calibration image. It detects the ArUco markers and then you manually place dots
on each color patch center. The result is saved as a new autolinearizer JSON file.

Usage:
    # Create from sample 006 calibration
    python scripts/create_autolinearizer.py --sample 006

    # Specify number of patches
    python scripts/create_autolinearizer.py --sample 006 --num-patches 28

    # Save to custom location
    python scripts/create_autolinearizer.py --sample 006 --output my_autolinearizer.json
"""

import argparse
import cv2
import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video2vision import utils as v2v_utils


def detect_aruco_markers(frame):
    """Detect ArUco markers in frame"""
    try:
        frame_float = frame.astype(np.float32) / 255.0
        frame_dict = {'image': frame_float[:, :, np.newaxis, :]}

        ts, corners = v2v_utils.locate_aruco_markers(frame_dict, np.array([0, 1, 2, 3]))

        if corners is not None and len(ts) > 0 and corners.shape[1] == 4:
            return corners[0], np.array([0, 1, 2, 3])
        return None, None
    except Exception as e:
        print(f"ArUco detection error: {e}")
        return None, None


class AutolinearizerBuilder:
    """Interactive tool for creating an autolinearizer"""

    def __init__(self, frame, corners, num_patches=28):
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.corners = corners  # (4, 4, 2) - 4 markers, 4 corners each
        self.num_patches = num_patches
        self.patch_positions = []
        self.window_name = "Create Autolinearizer - Click on patch centers"
        self.result = None

        # Calculate display scale
        h, w = self.frame.shape[:2]
        max_height = 1080
        if h > max_height:
            self.scale = max_height / h
        else:
            self.scale = 1.0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Transform mouse coordinates from display to original image
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)

            # Add patch
            if len(self.patch_positions) < self.num_patches:
                self.patch_positions.append([orig_x, orig_y])
                self.redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last patch
            if self.patch_positions:
                self.patch_positions.pop()
                self.redraw()

    def redraw(self):
        """Redraw the display with current patch positions"""
        self.display_frame = self.frame.copy()

        # Draw ArUco markers
        for marker_idx in range(4):
            marker_corners = self.corners[marker_idx].astype(np.int32)
            cv2.polylines(self.display_frame, [marker_corners], True, (0, 255, 0), 3)
            center = marker_corners.mean(axis=0).astype(np.int32)
            cv2.putText(self.display_frame, f"ID {marker_idx}", tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Draw patches
        for idx, (px, py) in enumerate(self.patch_positions):
            color = (0, 255, 255)  # Yellow for regular patches
            cv2.circle(self.display_frame, (px, py), 15, color, -1)
            cv2.putText(self.display_frame, str(idx), (px + 20, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw instructions
        remaining = self.num_patches - len(self.patch_positions)
        status = f"Patches: {len(self.patch_positions)}/{self.num_patches}"
        if remaining > 0:
            status += f" - Click {remaining} more"
        else:
            status += " - Press SPACE to save, R to reset"

        cv2.putText(self.display_frame, status, (30, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(self.display_frame, "Left Click: Add | Right Click: Remove Last | R: Reset | Space: Save | Q: Quit",
                   (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Scale for display if needed
        h, w = self.display_frame.shape[:2]
        max_height = 1080
        if h > max_height:
            scale = max_height / h
            display = cv2.resize(self.display_frame, None, fx=scale, fy=scale)
        else:
            display = self.display_frame

        cv2.imshow(self.window_name, display)

    def run(self):
        """Run the interactive patch selection"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.redraw()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Quit
                self.result = None
                break
            elif key == ord('r'):
                # Reset
                self.patch_positions = []
                self.redraw()
            elif key == ord(' '):
                # Save (space bar)
                if len(self.patch_positions) == self.num_patches:
                    self.result = self.patch_positions
                    break
                else:
                    print(f"Need {self.num_patches} patches, have {len(self.patch_positions)}")

        cv2.destroyAllWindows()
        return self.result


def main():
    parser = argparse.ArgumentParser(
        description="Create a custom autolinearizer from a calibration frame"
    )
    parser.add_argument('--sample', required=True, help='Sample ID (e.g., 006)')
    parser.add_argument('--num-patches', type=int, default=28,
                       help='Number of color patches (default: 28)')
    parser.add_argument('--output', default='data/autolinearizer_custom.json',
                       help='Output path for autolinearizer JSON (default: data/autolinearizer_custom.json)')

    args = parser.parse_args()

    # Load calibration frame
    sample_dir = Path('videos/samples') / args.sample
    cal_dir = sample_dir / 'calibration'

    if not cal_dir.exists():
        print(f"Error: Calibration directory not found: {cal_dir}")
        sys.exit(1)

    vis_videos = sorted(cal_dir.glob('VIS_*.MP4')) + sorted(cal_dir.glob('VIS_*.mp4'))
    if not vis_videos:
        print(f"Error: No VIS calibration video found in {cal_dir}")
        sys.exit(1)

    vis_path = str(vis_videos[0])
    print(f"Loading calibration frame from: {vis_path}")

    # Load first frame
    cap = cv2.VideoCapture(vis_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read video frame")
        sys.exit(1)

    print(f"Frame loaded: {frame.shape}")

    # Detect ArUco markers
    print("Detecting ArUco markers...")
    corners, ids = detect_aruco_markers(frame)

    if corners is None or corners.shape[0] != 4:
        print(f"Error: Could not detect all 4 ArUco markers")
        print(f"Found: {corners.shape[0] if corners is not None else 0}")
        sys.exit(1)

    print(f"✓ Detected 4 ArUco markers")

    # Run interactive patch placement
    print(f"\nInstructions:")
    print(f"  - Click on the CENTER of each of the {args.num_patches} color patches")
    print(f"  - Left click: Add patch")
    print(f"  - Right click: Remove last patch")
    print(f"  - R: Reset all patches")
    print(f"  - Space: Save when done")
    print(f"  - Q: Quit without saving")
    print()

    builder = AutolinearizerBuilder(frame, corners, args.num_patches)
    patch_positions = builder.run()

    if patch_positions is None:
        print("Cancelled by user")
        sys.exit(0)

    # Load expected values from the calibration CSV
    # This assumes you're using the same color chart as the default autolinearizer
    import csv

    calibration_csv = Path('data/aruco_samples.csv')
    camera_csv = Path('data/camera_sensitivities.csv')

    if calibration_csv.exists() and camera_csv.exists():
        # Load reflectance values (skip header row)
        sample_ref = np.loadtxt(calibration_csv, delimiter=',', skiprows=1)
        # Load camera sensitivities (skip header row)
        camera_sense = np.loadtxt(camera_csv, delimiter=',', skiprows=1)
        camera_sense = camera_sense / camera_sense.sum(axis=0, keepdims=True)

        # Calculate expected values: reflectance * camera_sensitivity
        expected_values = sample_ref.T.dot(camera_sense)

        # Limit to number of patches we have
        expected_values = expected_values[:len(patch_positions)]
    else:
        print(f"\nWarning: Could not load calibration data, using placeholder values")
        # Use placeholder values (will need to be updated)
        expected_values = [[0.5, 0.5, 0.5, 0.5]] * len(patch_positions)

    # Create autolinearizer data structure
    autolinearizer_data = {
        "class": "AutoLinearize",
        "marker_ids": [0, 1, 2, 3],
        "marker_points": corners.tolist(),  # (4, 4, 2)
        "sample_points": patch_positions,   # List of [x, y]
        "expected_values": expected_values.tolist(),
        "method": "poly",
        "order": 3,
        "sample_width": 25
    }

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(autolinearizer_data, f)

    print(f"\n✓ Autolinearizer saved to: {output_path}")
    print(f"\nTo use this autolinearizer:")
    print(f"  1. Update step2_extract_calibration.py to use '{output_path}'")
    print(f"  OR")
    print(f"  2. Replace data/autolinearizer.json with this file")
    print(f"\nNote: This autolinearizer uses the marker and patch POSITIONS only.")
    print(f"      It still uses the expected_values from data/aruco_samples.csv")


if __name__ == '__main__':
    main()
