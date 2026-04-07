# Batch Processing Scripts

Process multispectral video pairs through alignment, calibration, and animal vision conversion.

## Setup

Install the package in development mode:

```bash
pip install -e .
```

Then run scripts from the project root using module syntax:

```bash
python -m scripts.step1a_select_flips --all
python -m scripts.step1b_run_alignments --all --save-preview
```

## Directory Structure

```
videos/samples/
    001/
        VIS_001.MP4
        UV_001.MP4
        calibration/
            VIS_001.MP4
            UV_001.MP4
    002/
        ...
data/
    camera_sensitivities.csv
    aruco_samples.csv
    animal_sensitivities/
    converters/
```

Run `python -m scripts.check_setup` to verify your setup.

## Workflow

### Step 1a: Select Flips (Manual)

```bash
python -m scripts.step1a_select_flips --all
```

Click to select the correct UV flip direction for each sample. Saves flip config to each sample's `config.json`.

### Step 1b: Calculate Alignments (Automated)

```bash
python -m scripts.step1b_run_alignments --all --save-preview
```

Runs AutoAlign on all video pairs. Options:

- `--main-method ecc|aruco|any` -- alignment method for main videos (default: any)
- `--main-motion-type euclidean|affine|homography` -- motion model (default: homography)
- `--rejected-only` -- reprocess only rejected samples
- `--preview-only` -- regenerate previews from saved alignments
- `--no-temporal` -- disable temporal alignment

### Step 1c: Review Alignments (Manual)

```bash
python -m scripts.step1c_review_alignments --all
```

View composite images and approve/reject each alignment (Y/N/S keys).

### Step 2: Extract Calibration (Semi-automated)

```bash
python -m scripts.step2_extract_calibration --all
```

Detects ArUco markers in calibration frames and extracts color patch values. Options:

- `--auto-only` -- skip manual patch selection
- `--num-patches 8|24` -- number of patches to extract
- `--frame-offset N` -- use a later frame if markers aren't visible in frame 0

### Step 3: Apply Full Pipeline (Automated)

```bash
python -m scripts.step3_apply_full_pipeline --approved-only
```

Applies alignment, linearization, and animal vision conversion. Options:

- `--animal NAME` -- animal vision model (e.g., `apis`, `avian`)
- `--aligned-only` -- output aligned videos without color science
- `--preview N` -- process only first N frames
- `--output-format mp4|mov` -- output format (default: mp4)

## Utility Scripts

- `check_setup.py` -- verify directory structure and dependencies
- `create_autolinearizer.py` -- manually create a custom autolinearizer
- `convert_alignment_to_pipeline.py` -- convert config.json alignment to pipeline JSON
- `debug_aruco_warp.py` -- debug ArUco marker detection

## Common Options

All step scripts accept:

- `--samples 001 006 012` -- process specific samples
- `--all` -- process all samples
- `--samples-dir /path/to/samples` -- custom samples directory (default: `videos/samples`)
