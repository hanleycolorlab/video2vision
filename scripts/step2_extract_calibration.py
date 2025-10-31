#!/usr/bin/env python3
"""
Step 2: Extract calibration patches from aligned calibration frames (MANUAL - QUICK).

This loads the first frame of calibration videos, applies alignment, detects
ArUco markers (if present), and extracts color patch pixel values for linearization.

Usage (from project root):
    # Using module syntax:
    python -m scripts.step2_extract_calibration --all
    python -m scripts.step2_extract_calibration --all --auto-only
    python -m scripts.step2_extract_calibration --samples 001 006 012

    # Or if video2vision is installed:
    python scripts/step2_extract_calibration.py --all
    python scripts/step2_extract_calibration.py --all --auto-only
    python scripts/step2_extract_calibration.py --samples 001 006 012

    # Force re-extraction
    python scripts/step2_extract_calibration.py --samples 001 --force

    # Specify different patch count (default: 24, some samples use 8)
    python scripts/step2_extract_calibration.py --all --num-patches 8
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from video2vision import utils as v2v_utils
from video2vision.warp import Warp
from video2vision.operators import HorizontalFlip, VerticalFlip
from video2vision import elementwise

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, analysis plots will be skipped")


def load_sample_config(sample_id, samples_dir="videos/samples"):
    """Load configuration for a sample"""
    config_path = Path(samples_dir) / sample_id / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, "r") as f:
        return json.load(f)


def save_sample_config(sample_id, config, samples_dir="videos/samples"):
    """Save configuration for a sample"""
    config_path = Path(samples_dir) / sample_id / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def find_calibration_videos(sample_dir):
    """Find VIS and UV calibration video pair"""
    cal_dir = Path(sample_dir) / "calibration"
    if not cal_dir.exists():
        return None, None

    vis_videos = sorted(cal_dir.glob("VIS_*.MP4")) + sorted(cal_dir.glob("VIS_*.mp4"))
    uv_videos = sorted(cal_dir.glob("UV_*.MP4")) + sorted(cal_dir.glob("UV_*.mp4"))

    if not vis_videos or not uv_videos:
        return None, None

    return str(vis_videos[0]), str(uv_videos[0])


def load_and_align_frame(vis_path, uv_path, alignment_params, frame_offset=0):
    """Load frame from videos and apply alignment to UV

    Args:
        vis_path: Path to VIS video
        uv_path: Path to UV video
        alignment_params: Alignment parameters dict
        frame_offset: Frame number to extract (default: 0 = first frame)

    Returns:
        vis_frame: Aligned VIS frame (H, W, 3) uint8
        uv_frame: Aligned UV frame (H, W, 3) uint8
        vis_float: VIS frame as float32 [0, 1]
        uv_float: UV frame as float32 [0, 1]
    """
    # Load frames at specified offset
    vis_cap = cv2.VideoCapture(vis_path)
    uv_cap = cv2.VideoCapture(uv_path)

    # Read frames sequentially to desired offset
    # (CAP_PROP_POS_FRAMES is unreliable with MP4 files)
    vis_frame = None
    uv_frame = None

    for i in range(frame_offset + 1):
        ret_vis, vis_frame = vis_cap.read()
        ret_uv, uv_frame = uv_cap.read()

        if not ret_vis or not ret_uv:
            vis_cap.release()
            uv_cap.release()
            raise RuntimeError(f"Failed to read frame {i} (only {i} frames available)")

    vis_cap.release()
    uv_cap.release()

    # Convert to float32 [0, 1]
    vis_float = vis_frame.astype(np.float32) / 255.0
    uv_float = uv_frame.astype(np.float32) / 255.0

    # Apply flip to UV if needed
    flip_type = alignment_params.get("flip", "none")
    if flip_type == "horizontal":
        uv_float = np.flip(uv_float, axis=1).copy()
        uv_frame = np.flip(uv_frame, axis=1).copy()
    elif flip_type == "vertical":
        uv_float = np.flip(uv_float, axis=0).copy()
        uv_frame = np.flip(uv_frame, axis=0).copy()

    # Apply homography to UV if available
    if (
        alignment_params.get("has_homography")
        and "homography_matrix" in alignment_params
    ):
        homography = np.array(alignment_params["homography_matrix"])
        height, width = vis_float.shape[:2]
        output_size = tuple(alignment_params.get("output_size", (width, height)))

        warp_op = Warp(homography, output_size=output_size)
        uv_aligned_dict = warp_op.apply({"image": uv_float})
        uv_float = uv_aligned_dict["image"]

        # Also warp the uint8 version for display
        uv_frame_float = uv_frame.astype(np.float32) / 255.0
        uv_frame_dict = warp_op.apply({"image": uv_frame_float})
        uv_frame = (np.clip(uv_frame_dict["image"], 0, 1) * 255).astype(np.uint8)

    # Convert back to uint8 for display
    vis_frame_out = (np.clip(vis_float, 0, 1) * 255).astype(np.uint8)
    uv_frame_out = (np.clip(uv_float, 0, 1) * 255).astype(np.uint8)

    return vis_frame_out, uv_frame_out, vis_float, uv_float


def detect_aruco_markers(frame):
    """Detect ArUco markers in frame

    Args:
        frame: numpy array (H, W, 3) in uint8 format

    Returns:
        corners: (4, 4, 2) array of corner positions for 4 markers, or None if detection fails
        ids: (4,) array of marker IDs [0, 1, 2, 3], or None
    """
    try:
        # locate_aruco_markers expects dict with 'image' key in float32 [0,1] format
        # Video format is (H, W, T, C) not (T, H, W, C)
        frame_float = frame.astype(np.float32) / 255.0
        # Add time dimension: from (H, W, C) to (H, W, 1, C)
        frame_dict = {"image": frame_float[:, :, np.newaxis, :]}

        ts, corners = v2v_utils.locate_aruco_markers(frame_dict, np.array([0, 1, 2, 3]))

        if corners is not None and len(ts) > 0 and corners.shape[1] == 4:
            # corners shape is (time, marker_id, corner_id, x_or_y)
            # We want (marker_id, corner_id, x_or_y) for the first (only) time index
            return corners[0], np.array([0, 1, 2, 3])
        return None, None
    except Exception as e:
        print(f"    ArUco detection error: {e}")
        return None, None


def estimate_patch_positions_from_aruco(
    corners,
    num_patches=None,
    autolinearizer_path="data/autolinearizer_custom.json",
):
    """Estimate color patch positions based on ArUco markers using autolinearizer

    Uses the pre-built autolinearizer to transform reference sample points
    based on detected ArUco marker positions via homography.

    Args:
        corners: (4, 4, 2) array of ArUco marker corners
        num_patches: Number of patches (default: None = use all from autolinearizer)
        autolinearizer_path: Path to autolinearizer JSON

    Returns:
        positions: List of (x, y) tuples for patch centers
    """
    from video2vision import operators

    # Try to load and use the autolinearizer
    try:
        auto_op = operators.load_operator(autolinearizer_path)

        # If num_patches not specified, use all samples from autolinearizer
        num_samples = len(auto_op.sample_points)
        if num_patches is None:
            num_patches = num_samples

        # Compute homography from reference to detected markers
        ref_points = np.array(auto_op.marker_points).reshape(-1, 2).astype(np.float32)
        det_points = corners.reshape(-1, 2).astype(np.float32)

        H, _ = cv2.findHomography(ref_points, det_points, cv2.RANSAC)

        # Transform sample points using homography
        ref_samples = np.array(auto_op.sample_points, dtype=np.float32)
        ones = np.ones((ref_samples.shape[0], 1), dtype=np.float32)
        ref_samples_h = np.hstack([ref_samples, ones])

        transformed = (H @ ref_samples_h.T).T
        transformed_samples = transformed[:, :2] / transformed[:, 2:3]

        # Use all samples unless limited
        num_to_use = min(num_patches, len(transformed_samples))
        positions = [(int(x), int(y)) for x, y in transformed_samples[:num_to_use]]
        return positions

    except Exception as e:
        print(f"      Warning: Autolinearizer failed ({e}), using simple grid")
        # Fallback to simple grid estimation

        # If num_patches is None, default to 24
        if num_patches is None:
            num_patches = 24

        marker_centers = corners.mean(axis=1)
        sorted_idx = np.lexsort((marker_centers[:, 0], marker_centers[:, 1]))
        markers = marker_centers[sorted_idx]

        if markers.shape[0] == 4:
            tl = markers[0] if markers[0, 0] < markers[1, 0] else markers[1]
            tr = markers[1] if markers[0, 0] < markers[1, 0] else markers[0]
            bl = markers[2] if markers[2, 0] < markers[3, 0] else markers[3]
            br = markers[3] if markers[2, 0] < markers[3, 0] else markers[2]

            rows, cols = (
                (4, 6)
                if num_patches == 24
                else (
                    (2, 4)
                    if num_patches == 8
                    else (int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
                )
            )

            positions = []
            for i in range(rows):
                for j in range(cols):
                    t_row = i / (rows - 1) if rows > 1 else 0.5
                    t_col = j / (cols - 1) if cols > 1 else 0.5
                    top = tl * (1 - t_col) + tr * t_col
                    bottom = bl * (1 - t_col) + br * t_col
                    pos = top * (1 - t_row) + bottom * t_row
                    positions.append((int(pos[0]), int(pos[1])))

            return positions[:num_patches]

        return None


class PatchSelector:
    """Interactive patch selection tool"""

    def __init__(self, vis_frame, uv_frame, num_patches=24, initial_positions=None):
        self.vis_frame = vis_frame
        self.uv_frame = uv_frame
        self.num_patches = num_patches
        self.positions = list(initial_positions) if initial_positions else []
        self.window_name = "Select Calibration Patches"
        self.mode = "verify" if initial_positions else "select"
        self.result = None

    def mouse_callback(self, event, x, y, flags, param):
        if self.mode == "verify":
            return  # No editing in verify mode initially

        if event == cv2.EVENT_LBUTTONDOWN:
            # Add patch
            self.positions.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last patch
            if self.positions:
                self.positions.pop()

    def draw_display(self):
        """Draw the display with patches and instructions"""
        # Create composite (side-by-side or overlay)
        h, w = self.vis_frame.shape[:2]
        display = np.hstack([self.vis_frame, self.uv_frame])

        # Draw patches on both sides
        for i, (x, y) in enumerate(self.positions):
            # VIS side
            cv2.circle(display, (x, y), 15, (0, 255, 0), 2)
            cv2.putText(
                display,
                str(i + 1),
                (x - 10, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # UV side (offset by width)
            cv2.circle(display, (x + w, y), 15, (0, 255, 0), 2)
            cv2.putText(
                display,
                str(i + 1),
                (x + w - 10, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        # Instructions overlay
        bar_height = 80
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], bar_height), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)

        if self.mode == "verify":
            text1 = f"Auto-detected {len(self.positions)}/{self.num_patches} patches"
            text2 = "[Y/SPACE]=Accept  [E]=Edit  [R]=Retry  [N]=Change count  [S]=Skip  [Q]=Quit"
        else:
            if self.num_patches is not None:
                text1 = f"Click to place patches: {len(self.positions)}/{self.num_patches}"
            else:
                text1 = f"Click to place patches: {len(self.positions)} (any amount)"
            text2 = "[Left Click]=Add  [Right Click]=Remove  [SPACE]=Done  [S]=Skip  [Q]=Quit"

        cv2.putText(
            display, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        cv2.putText(
            display, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
        )

        return display

    def run(self):
        """Run the interactive selection"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            display = self.draw_display()
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if self.mode == "verify":
                if key == ord("y") or key == ord("Y") or key == ord(" "):
                    self.result = "accept"
                    break
                elif key == ord("e") or key == ord("E"):
                    self.mode = "select"
                elif key == ord("r") or key == ord("R"):
                    self.result = "retry"
                    break
                elif key == ord("n") or key == ord("N"):
                    self.result = "change_count"
                    break
                elif key == ord("s") or key == ord("S"):
                    self.result = "skip"
                    break
                elif key == ord("q") or key == ord("Q") or key == 27:
                    self.result = "quit"
                    break
            else:  # select mode
                if key == ord(" "):
                    # Accept if we have at least 1 patch, or if num_patches is set and we have enough
                    if len(self.positions) > 0 and (self.num_patches is None or len(self.positions) >= self.num_patches):
                        self.result = "accept"
                        break
                    elif len(self.positions) == 0:
                        print("    Need at least 1 patch")
                    else:
                        print(f"    Need at least {self.num_patches} patches, have {len(self.positions)}")
                elif key == ord("s") or key == ord("S"):
                    self.result = "skip"
                    break
                elif key == ord("q") or key == ord("Q") or key == 27:
                    self.result = "quit"
                    break

        cv2.destroyWindow(self.window_name)
        return self.result, self.positions


def extract_patch_values(frame_float, positions, patch_size=5):
    """Extract average pixel values from patches

    Args:
        frame_float: (H, W, C) float32 array [0, 1]
        positions: List of (x, y) patch centers
        patch_size: Radius of patch to average

    Returns:
        values: List of dicts with channel values
    """
    values = []
    h, w, c = frame_float.shape

    for x, y in positions:
        # Extract patch region
        x1 = max(0, x - patch_size)
        x2 = min(w, x + patch_size)
        y1 = max(0, y - patch_size)
        y2 = min(h, y + patch_size)

        patch = frame_float[y1:y2, x1:x2, :]

        # Average over patch
        avg = patch.mean(axis=(0, 1))

        # Store as dict
        if c == 3:
            values.append({"b": float(avg[0]), "g": float(avg[1]), "r": float(avg[2])})
        else:
            values.append({"value": float(avg.mean())})

    return values


def load_csv(path, normalize=False, skip_wavelength=False):
    """Load CSV file as numpy array

    Args:
        path: Path to CSV file
        normalize: If True, normalize each column
        skip_wavelength: If True, skip first column (assumes it's wavelength metadata)
    """
    data = np.loadtxt(path, delimiter=',', skiprows=1)  # Skip header row
    if skip_wavelength:
        # Skip first column (wavelength)
        data = data[:, 1:]
    if normalize:
        # Normalize each column
        data = data / data.sum(axis=0, keepdims=True)
    return data


def generate_calibration_analysis(sample_id, config, calibration_patches,
                                   calibration_values_path, camera_path,
                                   samples_dir='videos/samples'):
    """Generate analysis plots and metrics for calibration quality

    Args:
        sample_id: Sample identifier
        config: Sample configuration dict
        calibration_patches: Calibration patch data from config
        calibration_values_path: Path to expected reflectance CSV
        camera_path: Path to camera sensitivities CSV
        samples_dir: Directory containing samples

    Returns:
        dict with metrics (mae, r2 per channel)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("    ⚠ Matplotlib not available, skipping plots")
        return None

    # Load expected values
    sample_ref = load_csv(calibration_values_path, skip_wavelength=True)
    camera_sense = load_csv(camera_path, normalize=True, skip_wavelength=True)
    expected_values = sample_ref.T.dot(camera_sense)

    # Extract measured values from calibration data
    vis_patches = np.array([
        [p['r'], p['g'], p['b']]
        for p in calibration_patches['patch_values_vis']
    ])
    uv_patches = np.array([
        [p['b'], p['g'], p['r']]  # UV stored as BGR
        for p in calibration_patches['patch_values_uv']
    ])

    # Combine: [UV_R, VIS_R, VIS_G, VIS_B]
    # UV channel 2 (R in BGR) contains UV light - matches notebook/step3
    measured_samples = np.concatenate((uv_patches[:, [2]], vis_patches), axis=1)

    # Truncate to match
    num_patches = min(len(measured_samples), len(expected_values))
    measured_samples = measured_samples[:num_patches]
    expected_values_truncated = expected_values[:num_patches]

    # Build linearizer using power law method
    try:
        line_op = elementwise.build_linearizer(
            measured_samples, expected_values_truncated, method='power'
        )
        linearized_values = line_op.apply_values(measured_samples)
    except Exception as e:
        print(f"    ⚠ Could not build linearizer for analysis: {e}")
        return None

    # Calculate metrics
    metrics = {}
    band_names = ['UV', 'Red', 'Green', 'Blue']

    for band in range(4):
        residuals = linearized_values[:, band] - expected_values_truncated[:, band]
        mae = np.mean(np.abs(residuals))

        # R² calculation
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((expected_values_truncated[:, band] -
                        expected_values_truncated[:, band].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        metrics[band_names[band]] = {'MAE': float(mae), 'R2': float(r2)}

    # Create output directory
    analysis_dir = Path(samples_dir) / sample_id / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: Measured vs Actual (camera response)
    plt.subplot(1, 3, 1)
    for band, band_name in enumerate(band_names):
        color = (band_name[0].lower() if band_name != 'UV' else 'purple')
        plt.scatter(
            measured_samples[:, band],
            expected_values_truncated[:, band],
            label=band_name,
            color=color,
            s=100,
            alpha=0.6
        )
    plt.xlabel('Measured Value', fontsize=12)
    plt.ylabel('Actual Value', fontsize=12)
    plt.title('Camera Response Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Measured vs Linearized (linearization curve)
    plt.subplot(1, 3, 2)
    for band, band_name in enumerate(band_names):
        color = (band_name[0].lower() if band_name != 'UV' else 'purple')
        plt.scatter(
            measured_samples[:, band],
            linearized_values[:, band],
            label=band_name,
            color=color,
            s=100,
            alpha=0.6
        )
    plt.xlabel('Measured Value', fontsize=12)
    plt.ylabel('Linearized Value', fontsize=12)
    plt.title('Linearization Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Actual vs Linearized (accuracy)
    plt.subplot(1, 3, 3)
    for band, band_name in enumerate(band_names):
        color = (band_name[0].lower() if band_name != 'UV' else 'purple')
        plt.scatter(
            expected_values_truncated[:, band],
            linearized_values[:, band],
            label=band_name,
            color=color,
            s=100,
            alpha=0.6
        )
    # Add diagonal line (perfect fit)
    all_vals = np.concatenate([expected_values_truncated.flatten(),
                                linearized_values.flatten()])
    min_val, max_val = all_vals.min(), all_vals.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=2)
    plt.xlabel('Actual Value', fontsize=12)
    plt.ylabel('Linearized Value', fontsize=12)
    plt.title('Linearization Accuracy', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = analysis_dir / 'linearization_quality.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics to JSON
    metrics_path = analysis_dir / 'calibration_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save metrics to text file for easy viewing
    metrics_txt_path = analysis_dir / 'calibration_metrics.txt'
    with open(metrics_txt_path, 'w') as f:
        f.write("Calibration Quality Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Sample: {sample_id}\n")
        f.write(f"Number of patches: {num_patches}\n\n")
        f.write(f"{'Band':<10} {'MAE':<12} {'R²':<12}\n")
        f.write("-" * 40 + "\n")
        for band_name, vals in metrics.items():
            f.write(f"{band_name:<10} {vals['MAE']:<12.6f} {vals['R2']:<12.6f}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Extract calibration patches from aligned calibration frames"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true", help="Process all samples with calibration"
    )
    group.add_argument("--samples", nargs="+", help="Specific samples to process")

    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Skip manual intervention if ArUco detection fails",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=None,
        help="Number of patches (default: auto from autolinearizer=28, or specify 8/24)",
    )
    parser.add_argument(
        "--autolinearizer",
        default="data/autolinearizer_custom.json",
        help="Path to autolinearizer JSON (default: data/autolinearizer_custom.json)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-extract even if patches already exist"
    )

    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Frame number to extract from calibration video (default: 0 = first frame)",
    )

    parser.add_argument(
        "--export-analysis",
        action="store_true",
        help="Generate analysis plots/metrics for samples (works with already-processed samples)",
    )

    parser.add_argument(
        "--calibration-csv",
        default="data/aruco_samples.csv",
        help="Path to calibration reflectance CSV for analysis (default: data/aruco_samples.csv)",
    )

    parser.add_argument(
        "--camera-csv",
        default="data/camera_sensitivities.csv",
        help="Path to camera sensitivities CSV for analysis (default: data/camera_sensitivities.csv)",
    )

    parser.add_argument(
        "--samples-dir",
        default="videos/samples",
        help="Directory containing samples (default: videos/samples)",
    )

    args = parser.parse_args()

    # Set samples directory
    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        sys.exit(1)

    # Find samples
    if args.all:
        samples = sorted(
            [
                d.name
                for d in samples_dir.iterdir()
                if d.is_dir() and (d / "config.json").exists()
            ]
        )
    else:
        samples = args.samples

    if not samples:
        print("Error: No samples to process")
        sys.exit(1)

    print("=" * 70)
    print("Step 2: Extract Calibration Patches")
    print("=" * 70)
    print(f"Samples: {len(samples)}")
    print(f"Expected patches per sample: {args.num_patches}")
    if args.auto_only:
        print("Mode: Auto-only (skip manual intervention)")
    print()

    results = []

    for i, sample_id in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] Sample {sample_id}")
        print("-" * 70)

        # Load config
        config = load_sample_config(sample_id, args.samples_dir)
        if config is None:
            print(f"  ⚠ No config.json found")
            results.append((sample_id, "SKIPPED", "No config"))
            continue

        # Check if already done
        if config.get("calibration_patches") and not args.force:
            print("  ✓ Patches already extracted")

            # If export-analysis mode, generate analysis and skip extraction
            if args.export_analysis:
                if MATPLOTLIB_AVAILABLE:
                    print("  Generating calibration analysis...", end=" ", flush=True)
                    try:
                        metrics = generate_calibration_analysis(
                            sample_id, config, config["calibration_patches"],
                            args.calibration_csv, args.camera_csv,
                            args.samples_dir
                        )
                        if metrics:
                            print("✓")
                            print(f"    Metrics: UV MAE={metrics['UV']['MAE']:.4f}, R²={metrics['UV']['R2']:.4f}")
                            results.append((sample_id, "SUCCESS", "Analysis generated"))
                        else:
                            print("⚠ Skipped")
                            results.append((sample_id, "SKIPPED", "Analysis skipped"))
                    except Exception as e:
                        print(f"✗ Error: {e}")
                        results.append((sample_id, "FAILED", f"Analysis error: {str(e)[:30]}"))
                else:
                    print("  ⚠ Matplotlib not available for analysis")
                    results.append((sample_id, "SKIPPED", "No matplotlib"))
            else:
                print("    (use --force to redo or --export-analysis to generate plots)")
                results.append((sample_id, "SKIPPED", "Already extracted"))
            continue

        # Check if has calibration
        if not config.get("has_calibration"):
            print(f"  ⊘ No calibration videos")
            results.append((sample_id, "SKIPPED", "No calibration"))
            continue

        # Check if calibration alignment exists
        alignment_cal = config.get("alignment_calibration")
        if not alignment_cal or not alignment_cal.get("has_homography"):
            print(f"  ⚠ No calibration alignment")
            print(
                f"    Run: python scripts/step1b_run_alignments.py --samples {sample_id}"
            )
            results.append((sample_id, "SKIPPED", "No alignment"))
            continue

        # Find calibration videos
        sample_dir = Path(samples_dir) / sample_id
        vis_path, uv_path = find_calibration_videos(sample_dir)

        if vis_path is None:
            print(f"  ✗ Calibration videos not found")
            results.append((sample_id, "FAILED", "Videos not found"))
            continue

        print("  Calibration frame:")
        if args.frame_offset > 0:
            print(f"    Loading frame {args.frame_offset}... ", end="", flush=True)
        else:
            print("    Loading first frame... ", end="", flush=True)

        try:
            vis_frame, uv_frame, vis_float, uv_float = load_and_align_frame(
                vis_path, uv_path, alignment_cal, args.frame_offset
            )
            print(f"✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append((sample_id, "FAILED", f"Load error: {str(e)[:30]}"))
            continue

        print(f"    Detecting ArUco markers... ", end="", flush=True)

        # Try ArUco detection
        corners, ids = detect_aruco_markers(vis_frame)

        if corners is not None and corners.shape[0] == 4:
            print(f"✓ Found 4 markers")

            # Estimate patch positions from ArUco
            patch_positions = estimate_patch_positions_from_aruco(
                corners, args.num_patches, args.autolinearizer
            )

            if patch_positions:
                print(f"    Auto-detected {len(patch_positions)} patch positions")

                # Show for verification
                selector = PatchSelector(
                    vis_frame, uv_frame, args.num_patches, patch_positions
                )
                result, positions = selector.run()

                if result == "quit":
                    print("\n✗ Quit by user")
                    break
                elif result == "skip":
                    print(f"  ⊘ Skipped by user")
                    results.append((sample_id, "SKIPPED", "User skip"))
                    continue
                elif result == "retry":
                    # TODO: Implement retry logic
                    print(f"  ⚠ Retry not yet implemented")
                    results.append((sample_id, "SKIPPED", "Retry requested"))
                    continue
                elif result == "change_count":
                    print(f"  ⚠ Change count not yet implemented")
                    results.append((sample_id, "SKIPPED", "Count change requested"))
                    continue
                elif result == "accept":
                    patch_positions = positions
                else:
                    print(f"  ✗ Unknown result: {result}")
                    results.append((sample_id, "FAILED", "Unknown result"))
                    continue
            else:
                print(f"    ✗ Failed to estimate patch positions from ArUco")
                if args.auto_only:
                    results.append((sample_id, "SKIPPED", "ArUco failed, auto-only"))
                    continue
                patch_positions = None
        else:
            print(
                f"✗ Only found {corners.shape[0] if corners is not None else 0}/4 markers"
            )

            if args.auto_only:
                print(f"    Skipping (auto-only mode)")
                results.append((sample_id, "SKIPPED", "ArUco failed, auto-only"))
                continue

            # Manual selection
            print(f"    Falling back to manual patch selection...")
            # Allow any number of patches in manual mode
            selector = PatchSelector(vis_frame, uv_frame, num_patches=None)
            result, positions = selector.run()

            if result == "quit":
                print("\n✗ Quit by user")
                break
            elif result == "skip":
                print(f"  ⊘ Skipped by user")
                results.append((sample_id, "SKIPPED", "User skip"))
                continue
            elif result == "accept":
                patch_positions = positions
            else:
                print(f"  ✗ Selection cancelled")
                results.append((sample_id, "FAILED", "Selection cancelled"))
                continue

        # Extract pixel values
        print(f"  ✓ Extracting pixel values from {len(patch_positions)} patches...")

        vis_values = extract_patch_values(vis_float, patch_positions)
        uv_values = extract_patch_values(uv_float, patch_positions)

        # Save to config
        config["calibration_patches"] = {
            "aruco_detected": corners is not None,
            "num_patches": len(patch_positions),
            "frame_offset": args.frame_offset,
            "patch_positions": [{"x": x, "y": y} for x, y in patch_positions],
            "patch_values_vis": vis_values,
            "patch_values_uv": uv_values,
        }

        if corners is not None:
            config["calibration_patches"]["aruco_corners"] = corners.tolist()

        save_sample_config(sample_id, config, args.samples_dir)

        # Show preview stats
        if vis_values:
            r_vals = [v["r"] for v in vis_values]
            g_vals = [v["g"] for v in vis_values]
            b_vals = [v["b"] for v in vis_values]
            print(f"  Preview stats:")
            print(
                f"    VIS: R=[{min(r_vals):.2f}-{max(r_vals):.2f}], "
                + f"G=[{min(g_vals):.2f}-{max(g_vals):.2f}], "
                + f"B=[{min(b_vals):.2f}-{max(b_vals):.2f}]"
            )

        # Generate analysis if requested
        if MATPLOTLIB_AVAILABLE:
            print("  Generating calibration analysis...", end=" ", flush=True)
            try:
                metrics = generate_calibration_analysis(
                    sample_id, config, config["calibration_patches"],
                    args.calibration_csv, args.camera_csv,
                    args.samples_dir
                )
                if metrics:
                    print("✓")
                    print(f"    Metrics: UV MAE={metrics['UV']['MAE']:.4f}, R²={metrics['UV']['R2']:.4f}")
                else:
                    print("⚠ Skipped")
            except Exception as e:
                print(f"✗ Error: {e}")

        results.append((sample_id, "SUCCESS", f"{len(patch_positions)} patches"))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success_count = sum(1 for _, status, _ in results if status == "SUCCESS")
    failed_count = sum(1 for _, status, _ in results if status == "FAILED")
    skipped_count = sum(1 for _, status, _ in results if status == "SKIPPED")

    for sample_id, status, detail in results:
        symbol = "✓" if status == "SUCCESS" else "✗" if status == "FAILED" else "⊘"
        print(f"{symbol} {sample_id}: {status} - {detail}")

    print(f"\nSuccess: {success_count}/{len(results)}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")

    if success_count > 0:
        print(f"\n✓ Patch data saved to config.json for {success_count} samples")
        print(f"\nNext step: Build linearization curves and apply full pipeline")


if __name__ == "__main__":
    main()
