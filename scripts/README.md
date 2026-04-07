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

## High Bit Depth Output

Step 3 supports 10-bit (and higher) video output for preserving
color depth from professional cameras (e.g. Sony S-Log footage).
Requires FFmpeg installed with codec support.

```bash
# 10-bit ProRes (large, visually lossless, best for editing)
python -m scripts.step3_apply_full_pipeline \
    --approved-only --preserve-bit-depth \
    --output-format mov --output-codec prores

# 10-bit HEVC (smaller files, good for archiving)
python -m scripts.step3_apply_full_pipeline \
    --approved-only --preserve-bit-depth \
    --output-format mp4 --output-codec hevc
```

Codec options: `auto` (default -- prores for .mov, hevc for .mp4),
`prores`, `hevc`, `h264`.

### Verifying output

```bash
# Check bit depth
ffprobe -v error -select_streams v:0 \
    -show_entries stream=pix_fmt \
    -of default=noprint_wrappers=1:nokey=1 output.mov
# Expected: yuv422p10le (ProRes) or yuv420p10le (HEVC)

# Check audio was included
ffprobe -v error -select_streams a:0 \
    -show_entries stream=codec_name \
    -of default=noprint_wrappers=1:nokey=1 output.mov
# Expected: aac
```

### Known limitations

- ProRes requires MOV container; HEVC/H.264 work best with MP4
- ProRes playback on Windows requires codec packs; native on
  macOS
- Not all FFmpeg builds include `libx265` with 10-bit support
- H.264 codec falls back to 8-bit (10-bit H.264 support varies)

### Troubleshooting

- **"ffmpeg: command not found"**: Install via `brew install
  ffmpeg` (macOS) or `apt install ffmpeg` (Ubuntu). Verify
  codec support with `ffmpeg -codecs | grep hevc`.
- **Wrong colors in output**: Ensure input data follows BGR
  channel order (OpenCV convention).
- **Audio missing**: Check source video has audio
  (`ffprobe -show_streams input.mp4`) and that
  `--preserve-bit-depth` is set (audio passthrough only works
  with the FFmpeg writer path).
- **Wrong frame rate**: FPS is auto-detected from the source
  video. If detection fails, it defaults to 23.976 fps.

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
