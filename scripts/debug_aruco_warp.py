#!/usr/bin/env python3
"""
Debug script to visualize ArUco marker warping and patch position estimation.

This script helps diagnose issues with the homography transformation from
reference marker positions to detected marker positions.

Usage (from project root):
    python -m scripts.debug_aruco_warp

    Or if video2vision is installed:
    python scripts/debug_aruco_warp.py
"""

import cv2
import numpy as np

from video2vision import operators


def visualize_aruco_warp(
    autolinearizer_path="data/autolinearizer_custom.json"
):
    """Visualize the reference marker and sample point positions"""

    try:
        auto_op = operators.load_operator(autolinearizer_path)
    except Exception as e:
        print(f"Error loading autolinearizer: {e}")
        return

    print(f"Loaded autolinearizer from: {autolinearizer_path}")
    print(f"Number of markers: {len(auto_op.marker_points)}")
    print(f"Number of sample points: {len(auto_op.sample_points)}")

    # Create a blank canvas to visualize reference points
    marker_points = np.array(auto_op.marker_points)
    sample_points = np.array(auto_op.sample_points)

    # Get bounds
    all_points = np.vstack([marker_points.reshape(-1, 2), sample_points])
    min_x, min_y = all_points.min(axis=0).astype(int)
    max_x, max_y = all_points.max(axis=0).astype(int)

    print("\nReference coordinate bounds:")
    print(f"  X: [{min_x}, {max_x}]")
    print(f"  Y: [{min_y}, {max_y}]")

    # Create canvas with some margin
    margin = 100
    canvas_w = max_x - min_x + 2 * margin
    canvas_h = max_y - min_y + 2 * margin

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Offset for drawing
    offset_x = margin - min_x
    offset_y = margin - min_y

    # Draw marker positions
    print("\nMarker points (reference):")
    for i, marker in enumerate(marker_points):
        print(f"  Marker {i}:")
        # Draw corners
        for j, (x, y) in enumerate(marker):
            x_draw = int(x + offset_x)
            y_draw = int(y + offset_y)
            print(f"    Corner {j}: ({x:.1f}, {y:.1f})")
            cv2.circle(canvas, (x_draw, y_draw), 5, (0, 0, 255), -1)

        # Draw marker center
        center = marker.mean(axis=0)
        cx = int(center[0] + offset_x)
        cy = int(center[1] + offset_y)
        cv2.circle(canvas, (cx, cy), 10, (255, 0, 0), 2)
        cv2.putText(
            canvas,
            str(i),
            (cx - 10, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        # Draw marker boundary
        corners_draw = marker + np.array([offset_x, offset_y])
        corners_draw = corners_draw.astype(np.int32)
        cv2.polylines(canvas, [corners_draw], True, (0, 0, 255), 2)

    # Draw sample points
    print("\nSample points (reference):")
    for i, (x, y) in enumerate(sample_points):
        x_draw = int(x + offset_x)
        y_draw = int(y + offset_y)
        if i < 10:  # Only print first 10
            print(f"  Sample {i}: ({x:.1f}, {y:.1f})")
        cv2.circle(canvas, (x_draw, y_draw), 8, (0, 255, 0), 2)
        cv2.putText(
            canvas,
            str(i + 1),
            (x_draw - 10, y_draw - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 128, 0),
            1,
        )

    if len(sample_points) > 10:
        print(f"  ... and {len(sample_points) - 10} more")

    # Display
    cv2.namedWindow("Reference Positions", cv2.WINDOW_NORMAL)
    cv2.imshow("Reference Positions", canvas)

    print("\nLegend:")
    print("  RED circles: ArUco marker corners")
    print("  RED polygons: ArUco marker boundaries")
    print("  BLUE circles with numbers: Marker centers (0-3)")
    print("  GREEN circles with numbers: Sample points (1-N)")
    print("\nPress any key to close...")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_homography_transform(
    autolinearizer_path="data/autolinearizer_custom.json"
):
    """Test homography transformation with simulated detected markers"""

    try:
        auto_op = operators.load_operator(autolinearizer_path)
    except Exception as e:
        print(f"Error loading autolinearizer: {e}")
        return

    print("\n" + "=" * 70)
    print("Testing Homography Transformation")
    print("=" * 70)

    # Use reference markers as both source and destination (identity test)
    ref_markers = np.array(auto_op.marker_points)

    print(f"\nReference marker shape: {ref_markers.shape}")
    print("Expected shape: (4, 4, 2) - 4 markers, 4 corners, 2 coords")

    # Test 1: Identity transformation
    print("\nTest 1: Identity transformation (reference -> reference)")
    ref_points = ref_markers.reshape(-1, 2).astype(np.float32)
    print(f"  Reshaped to: {ref_points.shape}")

    H, _ = cv2.findHomography(ref_points, ref_points, cv2.RANSAC)
    print("  Homography matrix:")
    print(H)
    print("  Should be close to identity matrix")

    # Transform sample points
    sample_points = np.array(auto_op.sample_points, dtype=np.float32)
    ones = np.ones((sample_points.shape[0], 1), dtype=np.float32)
    sample_points_h = np.hstack([sample_points, ones])

    transformed = (H @ sample_points_h.T).T
    transformed_samples = transformed[:, :2] / transformed[:, 2:3]

    # Check error
    error = np.abs(transformed_samples - sample_points).max()
    print(f"  Max transformation error: {error:.4f} pixels")
    print("  (Should be near zero for identity)")

    # Test 2: Simple translation
    print("\nTest 2: Translation transformation (+100, +50)")
    det_points = ref_points + np.array([100.0, 50.0])

    H, _ = cv2.findHomography(ref_points, det_points, cv2.RANSAC)

    transformed = (H @ sample_points_h.T).T
    transformed_samples = transformed[:, :2] / transformed[:, 2:3]

    expected_translation = sample_points + np.array([100.0, 50.0])
    error = np.abs(transformed_samples - expected_translation).max()
    print(f"  Max transformation error: {error:.4f} pixels")
    print("  (Should be near zero for pure translation)")

    print("\n  Sample point 0:")
    print(f"    Original: {sample_points[0]}")
    print(f"    Transformed: {transformed_samples[0]}")
    print(f"    Expected: {expected_translation[0]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug ArUco warping")
    parser.add_argument(
        "--autolinearizer",
        default="data/autolinearizer_custom.json",
        help="Path to autolinearizer JSON",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization of reference positions",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run homography transformation tests"
    )

    args = parser.parse_args()

    if not args.visualize and not args.test:
        # Default: do both
        args.visualize = True
        args.test = True

    if args.visualize:
        visualize_aruco_warp(args.autolinearizer)

    if args.test:
        test_homography_transform(args.autolinearizer)
