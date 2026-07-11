# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

video2vision is a Python image processing toolkit for using multispectral videos to approximate the vision of animals. The toolkit provides support for aligning videos from separate cameras, linearizing to remove camera post-processing, and converting to animal vision.

## Development Setup

### Installation

Install the package and its dependencies:

```bash
python3 -m pip install .
python3 -m pip install -r requirements-optional.txt
```

### Core Dependencies

- **Required**: networkx, numpy, opencv-contrib-python, scipy, tdigest
- **Optional**: ipyevents, matplotlib, notebook, rawpy, sphinx, tabulate, tifffile

## Testing

### Running All Tests

```bash
python3 tests/run_all_tests.py
```

### Running Individual Test Files

```bash
python3 -m unittest tests.auto_operator_tests
python3 -m unittest tests.elementwise_tests
python3 -m unittest tests.io_tests
python3 -m unittest tests.operator_tests
python3 -m unittest tests.pipeline_tests
python3 -m unittest tests.utils_tests
python3 -m unittest tests.warp_tests
```

Note: Tests are configured to raise all warnings as errors.

## Architecture

### Core Concepts

video2vision is built around a **graph-based pipeline architecture** where image processing operations are composed as directed acyclic graphs (DAGs):

1. **Operators**: Base class for all image processing operations. Each operator transforms input images/videos and can be serialized to JSON. Operators work with dictionaries containing `'image'` (numpy.ndarray in float32, range [0,1]) and optional `'mask'` (uint8 boolean mask) keys.

2. **Pipeline**: Subclasses `networkx.DiGraph` to represent a directed graph of operators. Each node stores an operator, and edges define how outputs flow between operators. Pipelines can be serialized to/from JSON files and execute in topological order.

3. **AutoOperators**: Special operators that use the first batch of data to calculate parameters (e.g., `AutoAlign` finds homography from first frames, `AutoLinearize` learns linearization curves). These raise `ResetPipeline` exception to reprocess inputs after parameter determination.

4. **I/O System**: `Loader` and `Writer` operators handle reading from and writing to disk (images, videos, RAW files, TIFF). They integrate seamlessly into pipelines as graph nodes.

### Key Module Organization

- **operators.py**: Base `Operator` class and standard operations (LinearMap, Resize, Pad, ToRNL, UBGRtoXYZ, etc.). All operators are registered in `OPERATOR_REGISTRY`.

- **auto_operators.py**: Operators that auto-configure from first batch (`AutoAlign`, `AutoLinearize`, `AutoTemporalAlign`).

- **pipeline.py**: `Pipeline` class that orchestrates operator execution using networkx DAG traversal.

- **io.py**: I/O operators (`Loader`, `Writer`) and convenience functions (`load`, `save`) that handle multiple formats including RAW and TIFF.

- **warp.py**: Geometric transformation operators (`Warp` base class).

- **elementwise.py**: Per-pixel operations and linearization utilities.

- **utils.py**: Utility functions for image coercion, ArUco marker detection, motion detection, and sample extraction.

### Image/Video Convention

- **Images**: numpy.ndarray in (H, W, C) arrangement - Height, Width, Channel
- **Videos**: numpy.ndarray in (H, W, T, C) arrangement - Height, Width, Time, Channel
- **Values**: All images are float32 in range [0, 1]
- **Masks**: Optional uint8 in (H, W) arrangement

Note: Be careful with indexing - the time dimension is the 3rd axis for videos, not the 1st.

### Pipeline Execution Flow

1. Operators are added to pipeline with `add_operator()` or `Pipeline.chain()`
2. Edges define data flow between operators via `add_edge(from_idx, to_idx, in_slot=N)`
3. `Pipeline.run()` executes in topological sort order:
   - `Loader` nodes feed inputs
   - Each operator processes when all inputs are ready
   - Outputs propagate to downstream operators
   - `Writer` nodes save results
4. `OutOfInputs` exception terminates the pipeline when loaders are exhausted
5. `ResetPipeline` exception resets all operators and re-runs from start (used by AutoOperators)

### Serialization

Pipelines and operators serialize to JSON via `_to_json()` method. When loading, the `OPERATOR_REGISTRY` maps class names back to classes. This enables:
- Saving trained alignment pipelines for reuse
- Reproducible image processing workflows
- Parameter sharing across multiple videos

## Jupyter Workflow

Users typically work with two main notebooks:

1. **Alignment-Pipeline-Builder.ipynb**: Build and save alignment pipeline (creates JSON file)
2. **Video-Analysis.ipynb**: Load alignment pipeline, process videos through linearization and animal vision conversion

Additional notebooks: AutoLinearizer-Builder.ipynb, Sense-Converter-Builder.ipynb

## Git Workflow

- Main branch: `main`
- Current working branch: `feat/alignment-updates`
