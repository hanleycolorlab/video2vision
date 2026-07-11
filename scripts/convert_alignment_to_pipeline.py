#!/usr/bin/env python3
"""
Convert alignment config.json to video_alignment_pipeline.json format.

This script converts the alignment parameters saved by step1b (new format)
into the pipeline JSON format expected by Video-Analysis.ipynb (old format).

Usage:
    # Convert main video alignment
    python scripts/convert_alignment_to_pipeline.py \
        --config /path/to/sample/config.json \
        --output data/video_alignment_pipeline.json

    # Convert calibration video alignment
    python scripts/convert_alignment_to_pipeline.py \
        --config /path/to/sample/config.json \
        --calibration \
        --output data/still_alignment_pipeline.json

    # Auto-detect from sample ID
    python scripts/convert_alignment_to_pipeline.py \
        --sample 002 \
        --samples-dir "~/Desktop/UV Selects/Processing/All"
"""

import argparse
import json
import sys
from pathlib import Path


def create_pipeline_from_alignment(
    alignment_data, flip_type, video_type="main"
):
    """
    Create pipeline JSON structure from alignment data.

    Args:
        alignment_data: Dict with 'homography_matrix', 'output_size',
            'temporal_shift', etc.
        flip_type: One of 'none', 'horizontal', 'vertical'
        video_type: 'main' or 'calibration'

    Returns:
        List of pipeline nodes in the old format
    """
    homography_matrix = alignment_data.get("homography_matrix")
    output_size = alignment_data.get("output_size")
    temporal_shift = alignment_data.get("temporal_shift", 0)

    if not homography_matrix or not output_size:
        raise ValueError(
            "Missing required alignment data: homography_matrix or output_size"
        )

    # Build pipeline nodes
    pipeline = []
    current_index = 0

    # Node 0: UV Loader
    uv_loader = {
        "index": current_index,
        "operator": {
            "class": "Loader",
            "path": None,
            "batch_size": 1,
            "expected_size": output_size
        },
        "edges": []
    }
    current_index += 1

    # Add flip operator if needed
    flip_index = None
    if flip_type == "horizontal":
        flip_index = current_index
        flip_op = {
            "index": flip_index,
            "operator": {
                "class": "HorizontalFlip"
            },
            "edges": []
        }
        uv_loader["edges"].append([flip_index, {"in_slot": 0}])
        pipeline.append(uv_loader)
        pipeline.append(flip_op)
        current_index += 1
    elif flip_type == "vertical":
        flip_index = current_index
        flip_op = {
            "index": flip_index,
            "operator": {
                "class": "VerticalFlip"
            },
            "edges": []
        }
        uv_loader["edges"].append([flip_index, {"in_slot": 0}])
        pipeline.append(uv_loader)
        pipeline.append(flip_op)
        current_index += 1
    else:
        # No flip
        pipeline.append(uv_loader)

    # Node: Warp (with pre-calculated homography)
    warp_index = current_index
    warp_op = {
        "index": warp_index,
        "operator": {
            "class": "Warp",
            "coe": homography_matrix,
            "output_size": output_size
        },
        "edges": []
    }

    # Connect flip or UV loader to warp
    if flip_index is not None:
        flip_op["edges"].append([warp_index, {"in_slot": 0}])
    else:
        uv_loader["edges"].append([warp_index, {"in_slot": 0}])

    pipeline.append(warp_op)
    current_index += 1

    # Node: VIS Loader
    vis_loader_index = current_index
    vis_loader = {
        "index": vis_loader_index,
        "operator": {
            "class": "Loader",
            "path": None,
            "batch_size": 1,
            "expected_size": output_size
        },
        "edges": []
    }
    pipeline.append(vis_loader)
    current_index += 1

    # Node: AutoAlign or AutoTemporalAlign
    align_index = current_index

    if (video_type == "main" and temporal_shift is not None and
            temporal_shift != 0):
        # Use AutoTemporalAlign for main videos with temporal shift
        # Note: The old format expects time_shift_range, not the actual shift
        # We set a range around the detected shift
        shift_range = [-10, 10]  # Default range used in original

        align_op = {
            "index": align_index,
            "operator": {
                "class": "AutoTemporalAlign",
                "time_shift_range": shift_range,
                "mask": [50, 50, -50, -50],  # Default mask
                "bands": [[0, 1, 2], []],
                "method": "ecc"
            },
            "edges": []
        }
    else:
        # Use AutoAlign for calibration or videos without temporal shift
        align_op = {
            "index": align_index,
            "operator": {
                "class": "AutoAlign",
                "num_votes": 4,
                # Default mask from still_alignment_pipeline
                "mask": [27, 16, 5730, 3094],
                "bands": [[0, 1, 2], []],
                "method": "ecc"
            },
            "edges": []
        }

    # Connect warp and VIS loader to alignment
    warp_op["edges"].append([align_index, {"in_slot": 0}])
    vis_loader["edges"].append([align_index, {"in_slot": 1}])

    pipeline.append(align_op)
    current_index += 1

    # Node: Writer
    writer_index = current_index
    writer_op = {
        "index": writer_index,
        "operator": {
            "class": "Writer",
            "path": None,
            "extension": "mp4" if video_type == "main" else "png"
        },
        "edges": []
    }

    align_op["edges"].append([writer_index, {"in_slot": 0}])
    pipeline.append(writer_op)

    return pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Convert alignment config to pipeline JSON format"
    )

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', help='Path to config.json file')
    group.add_argument(
        '--sample',
        help='Sample ID (will look for config.json in samples-dir)'
    )

    # Options
    parser.add_argument(
        '--samples-dir',
        help='Directory containing samples (when using --sample)'
    )
    parser.add_argument(
        '--calibration', action='store_true',
        help='Convert calibration alignment instead of main'
    )
    parser.add_argument('--output', help='Output pipeline JSON file path')

    args = parser.parse_args()

    # Determine config path
    if args.sample:
        if not args.samples_dir:
            print("Error: --samples-dir required when using --sample")
            sys.exit(1)

        samples_dir = Path(args.samples_dir).expanduser()
        config_path = samples_dir / args.sample / "config.json"
    else:
        config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Determine which alignment to use
    if args.calibration:
        alignment_key = "alignment_calibration"
        flip_key = "flip_calibration"
        video_type = "calibration"
        default_output = "data/still_alignment_pipeline.json"
    else:
        alignment_key = "alignment_main"
        flip_key = "flip_main"
        video_type = "main"
        default_output = "data/video_alignment_pipeline.json"

    # Get alignment data
    alignment_data = config.get(alignment_key)
    if not alignment_data:
        print(f"Error: No {alignment_key} found in config")
        sys.exit(1)

    flip_type = config.get(flip_key, "none")

    # Check if approved
    review_status = alignment_data.get("review_status")
    if review_status == "rejected":
        print("Warning: Alignment was rejected in review")
        response = input("Continue anyway? [y/N] ")
        if response.lower() != 'y':
            print("Aborted")
            sys.exit(1)
    elif review_status == "approved":
        print("✓ Alignment approved in review")
    else:
        print("⚠ Alignment has not been reviewed yet")

    # Create pipeline
    try:
        pipeline = create_pipeline_from_alignment(
            alignment_data, flip_type, video_type
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Determine output path
    output_path = Path(args.output) if args.output else Path(default_output)

    # Save pipeline
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(pipeline, f, indent="\t")

    print(f"\n✓ Pipeline created: {output_path}")
    print("\nDetails:")
    print(f"  Sample: {config.get('sample_id', 'unknown')}")
    print(f"  Type: {video_type}")
    print(f"  Flip: {flip_type}")
    print(f"  Output size: {alignment_data.get('output_size')}")
    print(f"  Temporal shift: {alignment_data.get('temporal_shift', 0)}")
    print(f"  Method: {alignment_data.get('method')}")
    print("\nYou can now use this pipeline in Video-Analysis.ipynb")


if __name__ == "__main__":
    main()
