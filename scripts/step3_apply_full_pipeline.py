#!/usr/bin/env python3
"""
Step 3: Apply full pipeline - alignment, linearization, and animal vision conversion (AUTOMATED - SLOW).

This reads alignment and calibration data from config.json files and applies the complete
pipeline to generate animal vision videos. This is a single-pass operation with no intermediate
video exports to avoid recompression.

Pipeline: Loader → Flip → Warp (alignment) → Linearizer → SenseConverter → Writer

Usage (from project root):
    # Using module syntax:
    python -m scripts.step3_apply_full_pipeline --approved-only
    python -m scripts.step3_apply_full_pipeline --approved-only --animal apis
    python -m scripts.step3_apply_full_pipeline --samples 001 006 012 --animal avian

    # Or if video2vision is installed:
    python scripts/step3_apply_full_pipeline.py --approved-only
    python scripts/step3_apply_full_pipeline.py --approved-only --animal apis
    python scripts/step3_apply_full_pipeline.py --samples 001 006 012 --animal avian

    # Process all samples (ignore review status)
    python scripts/step3_apply_full_pipeline.py --all --animal bombus_terrestris_dalmaticus

    # Output aligned videos only (no color science transformations)
    python scripts/step3_apply_full_pipeline.py --approved-only --aligned-only

    # Preview mode with aligned-only
    python scripts/step3_apply_full_pipeline.py --samples 001 --aligned-only --preview 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from video2vision import io, pipeline, operators, elementwise, auto_operators
from video2vision.operators import HorizontalFlip, VerticalFlip, ConcatenateOnBands
from video2vision.warp import Warp
from video2vision.auto_operators import AutoTemporalAlign
from video2vision.io import OutOfInputs

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, analysis plots will be skipped")


class PreviewLoader(io.Loader):
    """Loader wrapper that limits the number of frames processed for preview mode.

    This wraps a standard Loader and stops yielding frames after max_frames,
    without requiring video re-encoding.
    """

    def __init__(self, path, expected_size, batch_size=1, num_channels=3, max_frames=None):
        """
        Args:
            path: Path to video/images
            expected_size: Expected (width, height)
            batch_size: Frames per batch
            num_channels: Number of channels
            max_frames: Maximum total frames to process (None = unlimited)
        """
        super().__init__(path, expected_size, batch_size, num_channels)
        self.max_frames = max_frames
        self.frames_yielded = 0

    def apply(self):
        """Override apply to stop after max_frames"""
        if self.max_frames is not None and self.frames_yielded >= self.max_frames:
            raise OutOfInputs("Preview frame limit reached")

        # Call parent's apply
        result = super().apply()

        # Track how many frames we've yielded
        if 'image' in result:
            # result['image'] is in HWTC format, T is axis 2
            if result['image'].ndim == 4:  # Video (HWTC)
                frames_in_batch = result['image'].shape[2]
            else:  # Image (HWC)
                frames_in_batch = 1

            self.frames_yielded += frames_in_batch

            # If we've exceeded max_frames, trim this batch
            if self.max_frames is not None and self.frames_yielded > self.max_frames:
                excess = self.frames_yielded - self.max_frames
                frames_to_keep = frames_in_batch - excess

                if result['image'].ndim == 4:
                    # Trim the time dimension
                    result['image'] = result['image'][:, :, :frames_to_keep, :]
                    if 'names' in result:
                        result['names'] = result['names'][:frames_to_keep]

                # Mark as final since we're stopping here
                result['final'] = True
                self.frames_yielded = self.max_frames

        return result


def load_pipeline_config(config_path=None):
    """Load global pipeline configuration

    Args:
        config_path: Optional path to config file (default: videos/samples/pipeline_config.json)
    """
    if config_path is None:
        config_path = Path("videos/samples/pipeline_config.json")
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"Error: Pipeline config not found at {config_path}")
        print("Please create pipeline_config.json with camera and animal settings.")
        sys.exit(1)

    with open(config_path, "r") as f:
        return json.load(f)


def load_sample_config(sample_id, samples_dir="videos/samples"):
    """Load configuration for a sample"""
    config_path = Path(samples_dir) / sample_id / "config.json"

    if not config_path.exists():
        return None

    with open(config_path, "r") as f:
        return json.load(f)


def load_csv(path, normalize=False, skip_wavelength=False):
    """Load CSV file as numpy array (matches notebook's load_csv logic)

    Args:
        path: Path to CSV file
        normalize: If True, normalize each column
        skip_wavelength: If True, skip first column (assumes it's wavelength metadata)
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)  # Skip header row
    if skip_wavelength:
        # Skip first column (wavelength)
        data = data[:, 1:]

    # Match notebook logic: convert from percentage if values > 2
    if data.max() > 2:
        data /= 100

    if normalize:
        # Normalize each column
        summand = data.sum(0, keepdims=True)
        if (np.abs(summand - 1) > 1e-2).any():
            print(f'Warning: Columns in {path} do not sum to 1: {summand.flatten()}. Normalizing.')
        data /= summand

    return data


def build_linearizer(calibration_patches, pipeline_config):
    """Build linearizer operator from calibration patch data

    Args:
        calibration_patches: Dict with 'patch_values_vis', 'patch_values_uv', 'num_patches'
        pipeline_config: Global pipeline configuration

    Returns:
        ElementwiseOperator for linearization
    """
    # Select calibration values path based on number of patches
    num_patches = calibration_patches.get("num_patches", 24)

    # Support both single path (legacy) and multiple paths
    calibration_config = pipeline_config.get("calibration_values_path")

    if isinstance(calibration_config, dict):
        # New format: multiple paths keyed by patch count
        # Convert string keys to integers for comparison
        patch_count_map = {int(k): v for k, v in calibration_config.items()}

        # Use exact match first, then fall back to closest available
        if num_patches in patch_count_map:
            calibration_values_path = patch_count_map[num_patches]
        else:
            # Fall back to closest available
            available_counts = sorted(patch_count_map.keys())
            if not available_counts:
                print("Error: No calibration paths defined in config")
                return None

            # Use the largest available that's <= num_patches, or smallest if all are larger
            suitable = [c for c in available_counts if c <= num_patches]
            if suitable:
                chosen = max(suitable)
            else:
                chosen = min(available_counts)

            calibration_values_path = patch_count_map[chosen]
            print(
                f"    Note: Using {chosen}-patch calibration for {num_patches} patches"
            )
    else:
        # Legacy format: single path string
        calibration_values_path = calibration_config

    camera_path = pipeline_config["camera_sensitivities_path"]

    print(f"    Loading calibration CSV: {calibration_values_path}")
    sample_ref = load_csv(calibration_values_path, skip_wavelength=True)
    camera_sense = load_csv(camera_path, normalize=True, skip_wavelength=True)

    # Calculate expected values: reflectance * camera_sensitivity
    expected_values = sample_ref.T.dot(camera_sense)
    print(f"    Expected values range: [{expected_values.min():.4f}, {expected_values.max():.4f}]")

    # Extract patch values from calibration data
    # Patches are stored as {"b": ..., "g": ..., "r": ...} from BGR videos
    vis_patches = np.array(
        [[p["b"], p["g"], p["r"]] for p in calibration_patches["patch_values_vis"]]
    )
    uv_patches = np.array(
        [[p["b"], p["g"], p["r"]] for p in calibration_patches["patch_values_uv"]]
    )

    # Combine: [UV_R, VIS_B, VIS_G, VIS_R]
    # UV channel 2 (R in BGR) contains UV light, VIS is full BGR
    samples = np.concatenate((uv_patches[:, [2]], vis_patches), axis=1)

    # Truncate to match number of patches (handle 8 vs 24 patch cases)
    num_patches = min(len(samples), len(expected_values))
    samples = samples[:num_patches]
    expected_values = expected_values[:num_patches]

    # Build linearizer based on camera type
    if pipeline_config.get("is_sony_camera", False):
        # Use Sony SLog3 PowerLaw function
        line_op = elementwise.PowerLaw(
            [
                [0.0047058172145495476, 4185.031519941784, -0.01, 0.16736099187966763],
                [0.0047058172145495476, 4185.031519941784, -0.01, 0.16736099187966763],
                [0.0047058172145495476, 4185.031519941784, -0.01, 0.16736099187966763],
                [0.0047058172145495476, 4185.031519941784, -0.01, 0.16736099187966763],
            ]
        )

        # Apply linearization and fit scale factors
        linearized_sample_values = line_op.apply_values(samples)
        for band in range(4):
            scale = np.linalg.lstsq(
                linearized_sample_values[:, [band]],
                expected_values[:, [band]],
                rcond=None,
            )[0][0]
            line_op.funcs[band].scale *= scale
            line_op.funcs[band].shift *= scale

        # Debug: check linearizer output range
        final_linearized = line_op.apply_values(samples)
        print(f"    Linearized output range: [{final_linearized.min():.4f}, {final_linearized.max():.4f}]")
        print(f"    Measured samples range: [{samples.min():.4f}, {samples.max():.4f}]")

    else:
        # Use polynomial fitting (power law)
        try:
            line_op = elementwise.build_linearizer(
                samples, expected_values, method="power"
            )
        except RuntimeError as e:
            print(f"    ✗ Linearization failed to converge: {e}")
            return None

    return line_op


def load_sense_converter(animal_type):
    """Load pre-built sense converter for animal vision

    Args:
        animal_type: Name of animal (e.g., 'apis', 'avian', 'bombus_terrestris_dalmaticus')

    Returns:
        Operator for converting linearized camera values to animal vision
    """
    converter_path = Path("data/converters") / f"{animal_type}_converter.json"

    if not converter_path.exists():
        print(f"Error: Sense converter not found: {converter_path}")
        print(f"\nAvailable converters:")
        converters_dir = Path("data/converters")
        if converters_dir.exists():
            for conv in sorted(converters_dir.glob("*_converter.json")):
                animal_name = conv.stem.replace("_converter", "")
                print(f"  - {animal_name}")
        return None

    return operators.load_operator(str(converter_path))


def find_video_pair(sample_dir):
    """Find VIS and UV video pair in main directory"""
    sample_dir = Path(sample_dir)

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


def copy_audio_to_video(source_video, target_video, output_video=None, audio_offset_frames=0, source_fps=None):
    """Copy audio from source video to target video using ffmpeg, with optional temporal offset

    Args:
        source_video: Path to video with audio
        target_video: Path to video without audio (will be replaced if output_video is None)
        output_video: Optional output path (if None, replaces target_video)
        audio_offset_frames: Number of frames to offset audio (positive = delay audio, negative = advance audio)
        source_fps: FPS of source video (required if audio_offset_frames != 0)

    Returns:
        True if successful, False otherwise
    """
    import subprocess
    import shutil

    if output_video is None:
        # Create temp file and replace original
        output_video = str(target_video) + ".tmp.mp4"
        replace_original = True
    else:
        replace_original = False

    try:
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(target_video),  # Video source (no audio)
            "-i", str(source_video),  # Audio source
            "-c:v", "copy",           # Copy video codec (no re-encode)
            "-c:a", "aac",            # Encode audio as AAC
            "-map", "0:v:0",          # Take video from first input
            "-map", "1:a:0?",         # Take audio from second input (? makes it optional)
        ]

        # Apply temporal offset if needed
        if audio_offset_frames != 0:
            if source_fps is None:
                raise ValueError("source_fps required when audio_offset_frames != 0")

            # Convert frame offset to seconds
            offset_seconds = audio_offset_frames / source_fps

            if offset_seconds > 0:
                # Positive offset: delay audio (add silence at start)
                cmd.extend(["-af", f"adelay={int(offset_seconds * 1000)}|{int(offset_seconds * 1000)}"])
            else:
                # Negative offset: advance audio (skip audio from start)
                # Note: we need to insert this BEFORE the audio input
                cmd.insert(3, "-ss")
                cmd.insert(4, str(abs(offset_seconds)))

        cmd.extend([
            "-shortest",              # Match shortest stream duration
            "-y",                     # Overwrite output
            str(output_video)
        ])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            # Audio might not exist in source, which is okay
            if "does not contain any stream" in result.stderr or "No such stream" in result.stderr:
                return False
            else:
                print(f"    Warning: ffmpeg error: {result.stderr[:100]}")
                return False

        if replace_original:
            # Replace original with audio-merged version
            shutil.move(output_video, target_video)

        return True

    except subprocess.TimeoutExpired:
        print("    Warning: ffmpeg timeout")
        return False
    except Exception as e:
        print(f"    Warning: Could not copy audio: {e}")
        return False


def trim_video(input_path, output_path, start_frame=0, max_frames=None):
    """Trim video starting from a specific frame

    Args:
        input_path: Path to input video
        output_path: Path to output video
        start_frame: Frame to start from (default: 0)
        max_frames: Maximum number of frames to write (default: None = all)

    Returns:
        Number of frames written
    """
    cap = cv2.VideoCapture(str(input_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = 0
    while True:
        if max_frames is not None and frame_count >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    return frame_count


def apply_alignment_only(
    vis_path,
    uv_path,
    alignment_params,
    aligned_vis_output_path,
    aligned_uv_output_path,
    batch_size=30,
    preview_frames=None,
):
    """Apply alignment transformation only (no color science)

    This outputs the original videos with only the alignment transformation applied.
    UV video gets flipped (if needed) and warped to match VIS dimensions.
    VIS video is output as-is (possibly trimmed if preview mode is active).

    Args:
        vis_path: Path to visible video
        uv_path: Path to UV video
        alignment_params: Dict with 'homography_matrix', 'flip', 'temporal_shift'
        aligned_vis_output_path: Where to save aligned VIS video
        aligned_uv_output_path: Where to save aligned UV video (warped)
        batch_size: Frames to process per batch
        preview_frames: If set, only process first N frames (preview mode)
    """
    # Get video dimensions from VIS (reference)
    cap_vis = cv2.VideoCapture(vis_path)
    vis_width = int(cap_vis.get(cv2.CAP_PROP_FRAME_WIDTH))
    vis_height = int(cap_vis.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_vis.release()

    # Check UV dimensions
    cap_uv = cv2.VideoCapture(uv_path)
    uv_width = int(cap_uv.get(cv2.CAP_PROP_FRAME_WIDTH))
    uv_height = int(cap_uv.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_uv.release()

    # Determine output size
    expected_size = (vis_width, vis_height)

    if alignment_params.get("output_size"):
        alignment_output = tuple(alignment_params["output_size"])
        if alignment_output != expected_size:
            print(
                f"Warning: Alignment output size {alignment_output} doesn't match VIS size {expected_size}"
            )
            expected_size = alignment_output
    elif alignment_params.get("homography_matrix"):
        if (uv_width, uv_height) != expected_size:
            raise ValueError(
                f"Video size mismatch: UV is {uv_width}x{uv_height} but VIS is {vis_width}x{vis_height}. "
                f"Run step1b again to regenerate alignment with correct output_size."
            )

    # Build pipeline for aligned outputs
    pipe = pipeline.Pipeline()

    # Loaders - use PreviewLoader if in preview mode
    if preview_frames is not None:
        vis_loader_idx = pipe.add_operator(
            PreviewLoader(vis_path, expected_size=expected_size, batch_size=batch_size, max_frames=preview_frames)
        )
        uv_loader_idx = pipe.add_operator(
            PreviewLoader(uv_path, expected_size=expected_size, batch_size=batch_size, max_frames=preview_frames)
        )
    else:
        vis_loader_idx = pipe.add_operator(
            io.Loader(vis_path, expected_size=expected_size, batch_size=batch_size)
        )
        uv_loader_idx = pipe.add_operator(
            io.Loader(uv_path, expected_size=expected_size, batch_size=batch_size)
        )

    # Apply flip to UV if needed
    current_uv_idx = uv_loader_idx
    flip_type = alignment_params.get("flip", "none")

    if flip_type == "horizontal":
        flip_idx = pipe.add_operator(HorizontalFlip())
        pipe.add_edge(uv_loader_idx, flip_idx, in_slot=0)
        current_uv_idx = flip_idx
    elif flip_type == "vertical":
        flip_idx = pipe.add_operator(VerticalFlip())
        pipe.add_edge(uv_loader_idx, flip_idx, in_slot=0)
        current_uv_idx = flip_idx

    # Apply homography warp to UV
    if alignment_params.get("homography_matrix"):
        homography = np.array(alignment_params["homography_matrix"])
        output_size = tuple(alignment_params.get("output_size", expected_size))

        warp_op = Warp(homography, output_size=output_size)
        warp_idx = pipe.add_operator(warp_op)
        pipe.add_edge(current_uv_idx, warp_idx, in_slot=0)
        current_uv_idx = warp_idx

    # Writer for aligned UV
    uv_writer_idx = pipe.add_operator(io.Writer(str(aligned_uv_output_path)))
    pipe.add_edge(current_uv_idx, uv_writer_idx, in_slot=0)

    # Writer for VIS (pass-through, just for temporal alignment if needed)
    vis_writer_idx = pipe.add_operator(io.Writer(str(aligned_vis_output_path)))
    pipe.add_edge(vis_loader_idx, vis_writer_idx, in_slot=0)

    # Run pipeline
    pipe.set_batch_size(batch_size)
    pipe.run()

    # Copy audio from original videos to aligned outputs
    print("    Copying audio... ", end="", flush=True)
    vis_audio_success = copy_audio_to_video(vis_path, aligned_vis_output_path)
    uv_audio_success = copy_audio_to_video(uv_path, aligned_uv_output_path)

    if vis_audio_success or uv_audio_success:
        audio_status = []
        if vis_audio_success:
            audio_status.append("VIS")
        if uv_audio_success:
            audio_status.append("UV")
        print(f"✓ ({', '.join(audio_status)})")
    else:
        print("⊘ (no audio tracks)")



def apply_full_pipeline(
    vis_path,
    uv_path,
    alignment_params,
    linearizer,
    sense_converter,
    animal_output_path,
    human_output_path,
    batch_size=30,
    preview_frames=None,
):
    """Apply complete pipeline: alignment → linearization → animal vision conversion

    This implementation matches the Video-Analysis.ipynb notebook's logic by using
    AutoTemporalAlign to handle temporal shift internally (without video re-encoding).

    Args:
        vis_path: Path to visible video
        uv_path: Path to UV video
        alignment_params: Dict with 'homography_matrix', 'flip', 'temporal_shift'
        linearizer: Linearization operator
        sense_converter: Animal vision conversion operator
        animal_output_path: Where to save animal vision video
        human_output_path: Where to save human vision video
        batch_size: Frames to process per batch
        preview_frames: If set, only process first N frames (preview mode)
    """
    # Get video dimensions from VIS (reference)
    cap_vis = cv2.VideoCapture(vis_path)
    vis_width = int(cap_vis.get(cv2.CAP_PROP_FRAME_WIDTH))
    vis_height = int(cap_vis.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_vis.release()

    # Check UV dimensions
    cap_uv = cv2.VideoCapture(uv_path)
    uv_width = int(cap_uv.get(cv2.CAP_PROP_FRAME_WIDTH))
    uv_height = int(cap_uv.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_uv.release()

    # Validate that alignment will produce correct output size
    expected_size = (vis_width, vis_height)

    if alignment_params.get("output_size"):
        alignment_output = tuple(alignment_params["output_size"])
        if alignment_output != expected_size:
            print(
                f"Warning: Alignment output size {alignment_output} doesn't match VIS size {expected_size}"
            )
            # Use alignment output size as the canonical size
            expected_size = alignment_output
    elif alignment_params.get("homography_matrix"):
        # If there's a homography but no explicit output_size, the warp will use UV dimensions
        # This could cause a mismatch if UV and VIS have different sizes
        if (uv_width, uv_height) != expected_size:
            print(
                f"Warning: UV size ({uv_width}x{uv_height}) != VIS size ({vis_width}x{vis_height})"
            )
            print(
                f"         Warp output will be ({uv_width}x{uv_height}), but VIS is ({vis_width}x{vis_height})"
            )
            raise ValueError(
                f"Video size mismatch: UV is {uv_width}x{uv_height} but VIS is {vis_width}x{vis_height}. "
                f"Run step1b again to regenerate alignment with correct output_size."
            )

    # Get temporal shift (manual override or from alignment)
    temporal_shift = alignment_params.get("temporal_shift", 0) or 0

    # Build pipeline - matching notebook architecture
    pipe = pipeline.Pipeline()

    # Loaders - use PreviewLoader if in preview mode
    if preview_frames is not None:
        vis_loader_idx = pipe.add_operator(
            PreviewLoader(vis_path, expected_size=expected_size, batch_size=batch_size, max_frames=preview_frames)
        )
        uv_loader_idx = pipe.add_operator(
            PreviewLoader(uv_path, expected_size=expected_size, batch_size=batch_size, max_frames=preview_frames)
        )
    else:
        vis_loader_idx = pipe.add_operator(
            io.Loader(vis_path, expected_size=expected_size, batch_size=batch_size)
        )
        uv_loader_idx = pipe.add_operator(
            io.Loader(uv_path, expected_size=expected_size, batch_size=batch_size)
        )

    # Apply flip to UV if needed (this matches the coarse warp in alignment pipeline)
    current_uv_idx = uv_loader_idx
    flip_type = alignment_params.get("flip", "none")

    if flip_type == "horizontal":
        flip_idx = pipe.add_operator(HorizontalFlip())
        pipe.add_edge(uv_loader_idx, flip_idx, in_slot=0)
        current_uv_idx = flip_idx
    elif flip_type == "vertical":
        flip_idx = pipe.add_operator(VerticalFlip())
        pipe.add_edge(uv_loader_idx, flip_idx, in_slot=0)
        current_uv_idx = flip_idx

    # Create AutoTemporalAlign operator with pre-set parameters
    # This matches how the notebook loads a saved alignment and applies it
    homography = np.array(alignment_params["homography_matrix"])
    output_size = tuple(alignment_params.get("output_size", expected_size))

    # Use AutoTemporalAlign even though we're not auto-detecting
    # This gives us the same temporal shift logic as the notebook
    align_op = AutoTemporalAlign(
        time_shift_range=[0, 0],  # Not used since we're setting time_shift directly
        bands=[[2], [0, 1, 2]],  # Extract UV_R + VIS_BGR (matches notebook line 194)
        coe=homography,  # Pre-set homography from alignment
        output_size=output_size,
        time_shift=temporal_shift,  # Manual temporal shift override
    )

    # Add alignment operator to pipeline
    align_idx = pipe.add_operator(align_op)
    pipe.add_edge(current_uv_idx, align_idx, in_slot=0)  # UV input
    pipe.add_edge(vis_loader_idx, align_idx, in_slot=1)  # VIS input

    # The AutoTemporalAlign operator outputs [UV_R, VIS_B, VIS_G, VIS_R]
    # This matches the notebook's architecture exactly

    # Apply linearization
    line_idx = pipe.add_operator(linearizer)
    pipe.add_edge(align_idx, line_idx, in_slot=0)

    # Apply animal vision conversion
    sense_idx = pipe.add_operator(sense_converter)
    pipe.add_edge(line_idx, sense_idx, in_slot=0)

    # Writer for animal vision
    animal_writer_idx = pipe.add_operator(io.Writer(str(animal_output_path)))
    pipe.add_edge(sense_idx, animal_writer_idx, in_slot=0)

    # Create human vision output (bands 1-3 from linearized data: VIS_B, VIS_G, VIS_R)
    # Linearized data is [UV_R, VIS_B, VIS_G, VIS_R], so bands 1-3 give BGR for OpenCV Writer
    human_sel_idx = pipe.add_operator(ConcatenateOnBands([[1, 2, 3]]))
    pipe.add_edge(line_idx, human_sel_idx, in_slot=0)

    # Writer for human vision
    human_writer_idx = pipe.add_operator(io.Writer(str(human_output_path)))
    pipe.add_edge(human_sel_idx, human_writer_idx, in_slot=0)

    # Run pipeline - batch size matches notebook (divided by 2 for full pipeline)
    pipe.set_batch_size(batch_size // 2)
    pipe.run()

    # Copy audio from VIS video to both outputs
    # The VIS video is the reference, so its audio timing is correct for the output
    # The temporal_shift was applied internally by AutoTemporalAlign, so the output
    # video frames are already synchronized with the VIS audio timeline
    print("    Copying audio... ", end="", flush=True)

    # Copy audio from VIS to both animal and human outputs
    # No offset needed because the pipeline output is already aligned to VIS timeline
    animal_audio_success = copy_audio_to_video(vis_path, animal_output_path)
    human_audio_success = copy_audio_to_video(vis_path, human_output_path)

    if animal_audio_success or human_audio_success:
        audio_status = []
        if animal_audio_success:
            audio_status.append("animal")
        if human_audio_success:
            audio_status.append("human")
        print(f"✓ ({', '.join(audio_status)})")
    else:
        print("⊘ (no audio track)")


def generate_ghosting_image(vis_path, uv_path, alignment_params, output_path):
    """Generate ghosting check image (R/B from VIS, G from aligned UV)

    Args:
        vis_path: Path to VIS video
        uv_path: Path to UV video
        alignment_params: Alignment parameters
        output_path: Where to save ghosting image
    """
    # Load first frames
    vis_cap = cv2.VideoCapture(vis_path)
    uv_cap = cv2.VideoCapture(uv_path)

    ret_vis, vis_frame = vis_cap.read()
    ret_uv, uv_frame = uv_cap.read()

    vis_cap.release()
    uv_cap.release()

    if not ret_vis or not ret_uv:
        return False

    # Convert to float
    vis_float = vis_frame.astype(np.float32) / 255.0
    uv_float = uv_frame.astype(np.float32) / 255.0

    # Apply flip to UV if needed
    flip_type = alignment_params.get("flip", "none")
    if flip_type == "horizontal":
        uv_float = np.flip(uv_float, axis=1).copy()
    elif flip_type == "vertical":
        uv_float = np.flip(uv_float, axis=0).copy()

    # Apply homography to UV
    if alignment_params.get("homography_matrix"):
        homography = np.array(alignment_params["homography_matrix"])
        height, width = vis_float.shape[:2]
        output_size = tuple(alignment_params.get("output_size", (width, height)))

        warp_op = Warp(homography, output_size=output_size)
        uv_aligned_dict = warp_op.apply({"image": uv_float})
        uv_aligned = uv_aligned_dict["image"]
    else:
        uv_aligned = uv_float

    # Create composite: R and B from VIS, G from UV
    # This makes misalignment visible as colored fringes
    composite = np.zeros_like(vis_float)
    composite[:, :, 0] = vis_float[:, :, 0]  # Blue from VIS
    composite[:, :, 1] = uv_aligned[:, :, 1]  # Green from UV
    composite[:, :, 2] = vis_float[:, :, 2]  # Red from VIS

    # Save
    composite_uint8 = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(output_path), composite_uint8)

    return True


def generate_pipeline_analysis(
    sample_id,
    config,
    calibration_patches,
    linearizer,
    sense_converter,
    animal_type,
    pipeline_config,
    output_dir,
    vis_path,
    uv_path,
    alignment_params,
):
    """Generate comprehensive analysis plots and metrics

    Args:
        sample_id: Sample identifier
        config: Sample configuration
        calibration_patches: Calibration patch data
        linearizer: Built linearizer operator
        sense_converter: Sense converter operator
        animal_type: Animal type string
        pipeline_config: Pipeline configuration dict
        output_dir: Output directory for analysis
        vis_path: Path to VIS video
        uv_path: Path to UV video
        alignment_params: Alignment parameters
    """
    if not MATPLOTLIB_AVAILABLE:
        print("    ⚠ Matplotlib not available, skipping analysis")
        return False

    analysis_dir = Path(output_dir) / sample_id / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate ghosting image for alignment check
    print("    - Ghosting image... ", end="", flush=True)
    try:
        ghosting_path = analysis_dir / "alignment_ghosting.png"
        if generate_ghosting_image(vis_path, uv_path, alignment_params, ghosting_path):
            print("✓")
        else:
            print("✗")
    except Exception as e:
        print(f"✗ ({e})")

    # 2. Linearization quality plots
    print("    - Linearization quality... ", end="", flush=True)
    try:
        # Get calibration path based on num_patches
        num_patches = calibration_patches.get("num_patches", 24)
        calibration_config = pipeline_config.get("calibration_values_path")

        if isinstance(calibration_config, dict):
            patch_count_map = {int(k): v for k, v in calibration_config.items()}
            if num_patches in patch_count_map:
                calibration_values_path = patch_count_map[num_patches]
            else:
                available_counts = sorted(patch_count_map.keys())
                suitable = [c for c in available_counts if c <= num_patches]
                chosen = max(suitable) if suitable else min(available_counts)
                calibration_values_path = patch_count_map[chosen]
        else:
            calibration_values_path = calibration_config

        camera_path = pipeline_config["camera_sensitivities_path"]

        # Load expected values
        sample_ref = load_csv(calibration_values_path, skip_wavelength=True)
        camera_sense = load_csv(camera_path, normalize=True, skip_wavelength=True)
        expected_values = sample_ref.T.dot(camera_sense)

        # Extract measured values
        vis_patches = np.array(
            [[p["b"], p["g"], p["r"]] for p in calibration_patches["patch_values_vis"]]
        )
        uv_patches = np.array(
            [[p["b"], p["g"], p["r"]] for p in calibration_patches["patch_values_uv"]]
        )

        measured_samples = np.concatenate((uv_patches[:, [2]], vis_patches), axis=1)

        num_patches_actual = min(len(measured_samples), len(expected_values))
        measured_samples = measured_samples[:num_patches_actual]
        expected_values_truncated = expected_values[:num_patches_actual]

        # Apply linearizer
        linearized_values = linearizer.apply_values(measured_samples)

        # Calculate metrics
        metrics = {}
        band_names = ["UV", "Red", "Green", "Blue"]

        for band in range(4):
            residuals = linearized_values[:, band] - expected_values_truncated[:, band]
            mae = np.mean(np.abs(residuals))
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum(
                (
                    expected_values_truncated[:, band]
                    - expected_values_truncated[:, band].mean()
                )
                ** 2
            )
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            metrics[band_names[band]] = {"MAE": float(mae), "R2": float(r2)}

        # Generate linearization plots
        fig = plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        for band, band_name in enumerate(band_names):
            color = band_name[0].lower() if band_name != "UV" else "purple"
            plt.scatter(
                measured_samples[:, band],
                expected_values_truncated[:, band],
                label=band_name,
                color=color,
                s=100,
                alpha=0.6,
            )
        plt.xlabel("Measured Value", fontsize=12)
        plt.ylabel("Actual Value", fontsize=12)
        plt.title("Camera Response Curve", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        for band, band_name in enumerate(band_names):
            color = band_name[0].lower() if band_name != "UV" else "purple"
            plt.scatter(
                measured_samples[:, band],
                linearized_values[:, band],
                label=band_name,
                color=color,
                s=100,
                alpha=0.6,
            )
        plt.xlabel("Measured Value", fontsize=12)
        plt.ylabel("Linearized Value", fontsize=12)
        plt.title("Linearization Curve", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        for band, band_name in enumerate(band_names):
            color = band_name[0].lower() if band_name != "UV" else "purple"
            plt.scatter(
                expected_values_truncated[:, band],
                linearized_values[:, band],
                label=band_name,
                color=color,
                s=100,
                alpha=0.6,
            )
        all_vals = np.concatenate(
            [expected_values_truncated.flatten(), linearized_values.flatten()]
        )
        min_val, max_val = all_vals.min(), all_vals.max()
        plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3, linewidth=2)
        plt.xlabel("Actual Value", fontsize=12)
        plt.ylabel("Linearized Value", fontsize=12)
        plt.title("Linearization Accuracy", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            analysis_dir / "linearization_quality.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Save metrics
        with open(analysis_dir / "linearization_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("✓")

    except Exception as e:
        print(f"✗ ({e})")

    # 3. Camera vs Animal sensitivity comparison
    print("    - Sensitivity comparison... ", end="", flush=True)
    try:
        camera_sense = load_csv(camera_path, normalize=True, skip_wavelength=True)

        # Get animal sensitivities
        animal_sense_path = (
            Path("data/animal_sensitivities") / f"{animal_type}_sensitivities.csv"
        )
        if animal_sense_path.exists():
            animal_sense = load_csv(str(animal_sense_path), normalize=True, skip_wavelength=True)

            fig = plt.figure(figsize=(15, 6))

            # Camera sensitivities
            plt.subplot(1, 2, 1)
            band_names = ["UV", "Blue", "Green", "Red"]
            n_camera_bands = camera_sense.shape[1]
            for band in range(n_camera_bands):
                band_name = band_names[band] if band < len(band_names) else f"Band {band+1}"
                color = band_name[0].lower() if band_name != "UV" else "purple"
                plt.plot(
                    np.arange(300, 701),
                    camera_sense[:, band],
                    color=color,
                    label=band_name,
                    linewidth=2,
                )
            plt.xlabel("Wavelength (nm)", fontsize=12)
            plt.ylabel("Sensitivity", fontsize=12)
            plt.title("Camera Spectral Sensitivity", fontsize=14, fontweight="bold")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Animal sensitivities
            plt.subplot(1, 2, 2)
            for band in range(animal_sense.shape[1]):
                plt.plot(
                    np.arange(300, 701),
                    animal_sense[:, band],
                    label=f"Photoreceptor {band+1}",
                    linewidth=2,
                )
            plt.xlabel("Wavelength (nm)", fontsize=12)
            plt.ylabel("Sensitivity", fontsize=12)
            plt.title(
                f'{animal_type.replace("_", " ").title()} Spectral Sensitivity',
                fontsize=14,
                fontweight="bold",
            )
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                analysis_dir / "sensitivity_comparison.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            print("✓")
        else:
            print(f"⚠ (no animal sensitivity file)")

    except Exception as e:
        print(f"✗ ({e})")

    # 4. Animal vision conversion accuracy
    print("    - Conversion accuracy... ", end="", flush=True)
    try:
        # Note: reusing sample_ref, measured_samples, and linearizer from earlier

        # Get animal sensitivities
        animal_sense_path = (
            Path("data/animal_sensitivities") / f"{animal_type}_sensitivities.csv"
        )
        if animal_sense_path.exists():
            animal_sense = load_csv(str(animal_sense_path), normalize=True, skip_wavelength=True)
            expected_animal_values = sample_ref.T.dot(animal_sense)

            # Get predicted values by running patches through full pipeline
            linearized_values = linearizer.apply_values(measured_samples)
            predicted_animal_values = linearized_values.dot(sense_converter.mat)

            # Truncate to same length
            min_len = min(len(expected_animal_values), len(predicted_animal_values))
            expected_animal_values = expected_animal_values[:min_len]
            predicted_animal_values = predicted_animal_values[:min_len]

            # Calculate metrics
            n_bands = animal_sense.shape[1]
            conversion_metrics = {}
            for band in range(n_bands):
                residuals = predicted_animal_values[:, band] - expected_animal_values[:, band]
                mae = np.mean(np.abs(residuals))
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum(
                    (expected_animal_values[:, band] - expected_animal_values[:, band].mean())**2
                )
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                conversion_metrics[f"Photoreceptor_{band+1}"] = {
                    "MAE": float(mae),
                    "R2": float(r2)
                }

            # Create plots - one subplot per photoreceptor
            _ = plt.figure(figsize=(6 * n_bands, 6))
            for band in range(n_bands):
                plt.subplot(1, n_bands, band + 1)
                # Use a colormap instead of trying to extract RGB
                scatter = plt.scatter(
                    expected_animal_values[:, band],
                    predicted_animal_values[:, band],
                    s=100,
                    c=np.arange(min_len),
                    cmap='viridis',
                    edgecolors='black',
                    alpha=0.7
                )
                plt.colorbar(scatter, label='Patch Index')
                plt.xlabel("Actual Animal Response", fontsize=12)
                plt.ylabel("Predicted Animal Response", fontsize=12)
                plt.title(
                    f"Photoreceptor {band+1}\nMAE={conversion_metrics[f'Photoreceptor_{band+1}']['MAE']:.2f}, "
                    f"R²={conversion_metrics[f'Photoreceptor_{band+1}']['R2']:.3f}",
                    fontsize=12,
                    fontweight="bold"
                )

                # Add diagonal line
                all_vals = np.concatenate([expected_animal_values[:, band], predicted_animal_values[:, band]])
                min_val, max_val = all_vals.min(), all_vals.max()
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=2)
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                analysis_dir / "conversion_accuracy.png",
                dpi=150,
                bbox_inches="tight"
            )
            plt.close()

            # Save conversion metrics
            with open(analysis_dir / "conversion_metrics.json", "w") as f:
                json.dump(conversion_metrics, f, indent=2)

            print("✓")
        else:
            print(f"⚠ (no animal sensitivity file)")

    except Exception as e:
        print(f"✗ ({e})")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Apply full pipeline: alignment + linearization + animal vision conversion"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true", help="Process all samples with required data"
    )
    group.add_argument(
        "--approved-only",
        action="store_true",
        help="Only process samples with approved alignments (recommended)",
    )
    group.add_argument("--samples", nargs="+", help="Process specific samples")

    parser.add_argument(
        "--animal",
        type=str,
        help="Animal type (e.g., apis, avian, bombus). Overrides pipeline_config.json",
    )
    parser.add_argument(
        "--pipeline-config",
        type=str,
        help="Path to pipeline config JSON (default: videos/samples/pipeline_config.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="videos/output",
        help="Output directory for final videos (default: videos/output)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Frames to process per batch (default: 32)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Reprocess even if output already exists"
    )

    parser.add_argument(
        "--samples-dir",
        default="videos/samples",
        help="Directory containing samples (default: videos/samples)",
    )
    parser.add_argument(
        "--save-analysis",
        action="store_true",
        help="Generate analysis plots and visualizations",
    )
    parser.add_argument(
        "--preview",
        type=int,
        nargs="?",
        const=30,
        metavar="FRAMES",
        help="Preview mode: process only first N frames (default: 30). "
        'Outputs will have "_preview" suffix.',
    )
    parser.add_argument(
        "--aligned-only",
        action="store_true",
        help="Output aligned videos only (no linearization or animal vision conversion). "
        "Produces VIS_aligned.mp4 and UV_aligned.mp4 for each sample.",
    )

    args = parser.parse_args()

    # Load global pipeline config
    pipeline_config = load_pipeline_config(args.pipeline_config)

    # Skip animal vision setup if aligned-only mode
    if args.aligned_only:
        animal_type = None
        sense_converter = None
    else:
        # Override animal type if specified
        if args.animal:
            pipeline_config["animal_type"] = args.animal

        # Validate animal type
        animal_type = pipeline_config.get("animal_type")
        if not animal_type:
            print(
                "Error: animal_type not specified in pipeline_config.json or --animal flag"
            )
            sys.exit(1)

        # Load sense converter
        sense_converter = load_sense_converter(animal_type)
        if sense_converter is None:
            sys.exit(1)

    # Set samples directory
    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        sys.exit(1)

    # Find samples
    if args.all or args.approved_only:
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
    if args.aligned_only:
        print("Step 3: Apply Alignment Only (No Color Science)")
    else:
        print("Step 3: Apply Full Pipeline (Alignment + Linearization + Animal Vision)")
    print("=" * 70)
    print(f"Samples: {len(samples)}")
    if args.aligned_only:
        print("Mode: Alignment only (no linearization or animal vision)")
    else:
        print(f"Animal type: {animal_type}")
        print(
            f"Camera: {'Sony SLog3' if pipeline_config.get('is_sony_camera') else 'Generic'}"
        )
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    if args.approved_only:
        print("Filter: Approved alignments only")
    if args.preview is not None:
        print(f"🎬 PREVIEW MODE: Processing first {args.preview} frames only")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, sample_id in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] Sample {sample_id}")
        print("-" * 70)

        # Load config
        config = load_sample_config(sample_id, args.samples_dir)
        if config is None:
            print(f"  ⚠ No config.json found, skipping")
            results.append((sample_id, "SKIPPED", "No config"))
            continue

        # Check for alignment
        alignment_main = config.get("alignment_main")
        if not alignment_main:
            print(f"  ⚠ No alignment data")
            print(
                f"    Run: python scripts/step1b_run_alignments.py --samples {sample_id}"
            )
            results.append((sample_id, "SKIPPED", "No alignment"))
            continue

        # Check review status if --approved-only
        if args.approved_only:
            review_status = alignment_main.get("review_status")
            if review_status != "approved":
                print(
                    f"  ⊘ Alignment not approved (status: {review_status or 'unreviewed'})"
                )
                results.append((sample_id, "SKIPPED", "Not approved"))
                continue

        # Check for calibration patches (only required if not aligned-only mode)
        if not args.aligned_only:
            calibration_patches = config.get("calibration_patches")
            if not calibration_patches:
                print("  ⚠ No calibration patch data")
                print(
                    f"    Run: python scripts/step2_extract_calibration.py --samples {sample_id}"
                )
                results.append((sample_id, "SKIPPED", "No calibration"))
                continue
        else:
            calibration_patches = None

        # Find video files
        sample_dir = Path(samples_dir) / sample_id
        vis_path, uv_path = find_video_pair(sample_dir)

        if vis_path is None:
            print(f"  ✗ Videos not found")
            results.append((sample_id, "FAILED", "Videos not found"))
            continue

        print(f"  VIS: {Path(vis_path).name}")
        print(f"  UV:  {Path(uv_path).name}")

        # Build linearizer (only if not aligned-only mode)
        if not args.aligned_only:
            print("  Building linearizer... ", end="", flush=True)
            try:
                linearizer = build_linearizer(calibration_patches, pipeline_config)
                if linearizer is None:
                    results.append((sample_id, "FAILED", "Linearizer build failed"))
                    continue
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
                results.append((sample_id, "FAILED", f"Linearizer: {str(e)[:50]}"))
                continue
        else:
            linearizer = None

        # Output paths
        sample_output_dir = output_dir / sample_id
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        # Add preview suffix if in preview mode
        preview_suffix = "_preview" if args.preview is not None else ""

        if args.aligned_only:
            # Aligned-only mode: output VIS_aligned and UV_aligned
            aligned_vis_output_path = sample_output_dir / f"{sample_id}_VIS_aligned{preview_suffix}.mp4"
            aligned_uv_output_path = sample_output_dir / f"{sample_id}_UV_aligned{preview_suffix}.mp4"

            # Check if already exists
            if (
                aligned_vis_output_path.exists()
                and aligned_uv_output_path.exists()
                and not args.force
            ):
                print("  ✓ Already processed")
                print(f"    VIS aligned: {aligned_vis_output_path.name}")
                print(f"    UV aligned:  {aligned_uv_output_path.name}")
                print("    (use --force to reprocess)")
                results.append((sample_id, "SUCCESS", "Already exists"))
                continue

            # Process alignment only
            print("  Applying alignment... ", end="", flush=True)
            start = time.time()

            try:
                apply_alignment_only(
                    vis_path,
                    uv_path,
                    alignment_main,
                    aligned_vis_output_path,
                    aligned_uv_output_path,
                    batch_size=args.batch_size,
                    preview_frames=args.preview,
                )
                elapsed = time.time() - start
                print(f"✓ Done in {elapsed:.1f}s")
                print("  Output:")
                print(f"    VIS aligned: {aligned_vis_output_path.name}")
                print(f"    UV aligned:  {aligned_uv_output_path.name}")
                results.append((sample_id, "SUCCESS", "Aligned"))
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
                results.append((sample_id, "FAILED", f"{str(e)[:50]}"))
                continue
        else:
            # Full pipeline mode: animal vision and human vision
            animal_output_path = (
                sample_output_dir / f"{sample_id}_animal_{animal_type}{preview_suffix}.mp4"
            )
            human_output_path = sample_output_dir / f"{sample_id}_human{preview_suffix}.mp4"

            # Check if already exists
            if (
                animal_output_path.exists()
                and human_output_path.exists()
                and not args.force
            ):
                print("  ✓ Already processed")
                print(f"    Animal: {animal_output_path.name}")
                print(f"    Human:  {human_output_path.name}")
                print("    (use --force to reprocess)")
                results.append((sample_id, "SUCCESS", "Already exists"))
                continue

            # Process full pipeline
            print("  Processing full pipeline... ", end="", flush=True)
            start = time.time()

            try:
                apply_full_pipeline(
                    vis_path,
                    uv_path,
                    alignment_main,
                    linearizer,
                    sense_converter,
                    animal_output_path,
                    human_output_path,
                    batch_size=args.batch_size,
                    preview_frames=args.preview,
                )
                elapsed = time.time() - start
                print(f"✓ Done in {elapsed:.1f}s")
                print("  Output:")
                print(f"    Animal: {animal_output_path.name}")
                print(f"    Human:  {human_output_path.name}")
                results.append((sample_id, "SUCCESS", "Processed"))

                # Generate analysis if requested
                if args.save_analysis:
                    print("  Generating analysis... ", end="", flush=True)
                    try:
                        generate_pipeline_analysis(
                            sample_id,
                            config,
                            calibration_patches,
                            linearizer,
                            sense_converter,
                            animal_type,
                            pipeline_config,
                            args.output_dir,
                            vis_path,
                            uv_path,
                            alignment_main,
                        )
                        print("✓")
                    except Exception as e:
                        print(f"⚠ Warning: Analysis generation failed: {e}")
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback

                traceback.print_exc()
                results.append((sample_id, "FAILED", f"{str(e)[:50]}"))
                continue

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
        if args.aligned_only:
            print(f"\n✓ Aligned videos saved to: {args.output_dir}")
            print("\nAll done! Your aligned VIS and UV videos are ready.")
        else:
            print(f"\n✓ Animal vision videos saved to: {args.output_dir}")
            print(f"\nAll done! Your {animal_type} vision videos are ready for analysis.")


if __name__ == "__main__":
    main()
