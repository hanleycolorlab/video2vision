# Video2Vision Batch Processing Scripts

Process multispectral video pairs efficiently by separating manual and automated steps.

## Quick Start

```bash
source .venv/bin/activate

# Step 1a: Select flips for all samples (10 min - click through)
python scripts/step1a_select_flips.py --all

# Step 1b: Calculate alignments + generate previews (~1-2 hours)
python scripts/step1b_run_alignments.py --all --save-preview

# Step 1c: Review alignment composites (~2-3 min)
python scripts/step1c_review_alignments.py --all

# Step 2: Extract calibration patches (~15 min)
python scripts/step2_extract_calibration.py --all

# Step 3: Apply full pipeline to main videos (overnight)
python scripts/step3_apply_full_pipeline.py --approved-only
```

**Using a different samples directory:**

```bash
# All scripts support --samples-dir argument
python scripts/step1a_select_flips.py --all --samples-dir /path/to/other/videos
python scripts/step3_apply_full_pipeline.py --approved-only --samples-dir /path/to/other/videos
```

## The Workflow

**Key principle:** Batch all manual work together, then run automation overnight.

### Environment Setup

Before running any scripts, you need to set up a Python virtual environment and install the required dependencies.

**Create and activate virtual environment:**

```bash
# Navigate to the video2vision directory
cd /path/to/video2vision

# Create virtual environment (Python 3.8+ required)
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
```

**Install dependencies:**

```bash
# Install core package and dependencies
python3 -m pip install .

# Install optional dependencies (required for these scripts)
python3 -m pip install -r requirements-optional.txt
```

**Verify installation:**

```bash
# Check that video2vision is installed
python -c "import video2vision; print('✓ video2vision installed')"

# Check optional dependencies
python -c "import matplotlib; import cv2; print('✓ Optional dependencies installed')"
```

**Note:** Always activate the virtual environment before running any scripts:

```bash
source .venv/bin/activate
```

### Directory Set Up

This set of scripts expects a specific set of directory structures to automate that batch processing. You will want a single parent directory (the path you will pass to the `--samples-dir` argument). Within that directory, you will create a subfolder for each pair of videos. I've found its easy just to names this 001, 002, etc. Within that folder you will have the main UV and Visible video files (naming conventions: UV videos start with UV* and likewise the Visible videos should start with VIS* ) and then a subfolder called `calibration` with the related calibration clips. So in all, each sample folder will look like the below. Throughout this full process, additional files and directories will be added to this this subfolder with sample outputs, configuration, analyses, etc.

```
your-samples-directory/
├── 001/
│   ├── UV_clip1.mp4          # Primary UV clip
│   ├── VIS_clip1.mp4         # Primary Visible clip
│   └── calibration/
│       ├── UV_clip2.mp4      # UV clip of calibration standard
│       └── VIS_clip2.mp4     # Visible clip of calibration standard
├── 002/
│   ├── UV_*.mp4
│   ├── VIS_*.mp4
│   └── calibration/
│       ├── UV_*.mp4
│       └── VIS_*.mp4
├── 003/
├── 004/
├── ...
└── 100/
```

### Step 1a: Select Flips (Manual - 10 min for 40 samples)

```bash
# Process all samples
python scripts/step1a_select_flips.py --all

# Process specific samples
python scripts/step1a_select_flips.py --samples 001 006 012

# Use different samples directory
python scripts/step1a_select_flips.py --all --samples-dir /path/to/videos
```

**What it does:**

- Shows 2x2 grid: VIS (reference) + 3 UV flip options
- Click the matching UV image (none, horizontal, vertical)
- Handles both main and calibration videos
- Saves to `videos/samples/{id}/config.json`

**Interactive controls:**

- **Left Click**: Select UV image that matches VIS orientation
- **S**: Skip this sample
- **Q**: Quit

**Output:** Per-sample `config.json` with flip settings

---

### Step 1b: Calculate Alignments (Automated - 1-2 hours)

```bash
# STANDARD WORKFLOW: Calculate alignment + save preview images
python scripts/step1b_run_alignments.py --all --save-preview

# Process specific samples
python scripts/step1b_run_alignments.py --samples 001 006 012 --save-preview

# PREVIEW-ONLY MODE: Regenerate previews from existing alignments
python scripts/step1b_run_alignments.py --all --preview-only

# Override alignment methods
python scripts/step1b_run_alignments.py --all --main-method ecc --calibration-method aruco

# Try different motion models (if alignment fails)
python scripts/step1b_run_alignments.py --all --save-preview --main-motion-type euclidean
python scripts/step1b_run_alignments.py --all --save-preview --main-motion-type affine

# Reprocess only rejected samples (after step1c review)
python scripts/step1b_run_alignments.py --all --rejected-only --save-preview --main-motion-type euclidean

# Disable temporal alignment (if cameras are perfectly synced)
python scripts/step1b_run_alignments.py --all --save-preview --no-temporal

# Process only main OR calibration videos
python scripts/step1b_run_alignments.py --all --main-only --save-preview
python scripts/step1b_run_alignments.py --all --calibration-only --save-preview

# Use a known-good alignment as starting point
python scripts/step1b_run_alignments.py --samples 001 --save-preview --save-as-template
python scripts/step1b_run_alignments.py --all --save-preview --use-initial-transform 001

# Force reprocessing
python scripts/step1b_run_alignments.py --samples 001 --save-preview --force
```

**What it does:**

- Uses first 30 frames to calculate alignment parameters
- Main videos: ECC alignment (~1-2 min/sample, no markers needed)
- Calibration videos: ArUco alignment (~30 sec/sample, requires markers)
- Saves homography matrix, temporal shift, and preview images
- Saves to `videos/samples/{id}/config.json`

**Alignment methods:**

- `--main-method` / `--calibration-method`:
  - **ecc** (default for main): Enhanced Correlation Coefficient, works without markers, ~1-2 min per sample
  - **aruco** (default for calibration): Requires ArUco markers (IDs: 0, 1, 2, 3), ~30 sec per sample
  - **any**: Try ArUco first, fall back to ECC if ArUco fails

**Motion models:**

- `--main-motion-type` / `--calibration-motion-type`:
  - **homography** (default): Full perspective transform with skew/warp (most flexible)
  - **affine**: Rotation + translation + scale + shear (no perspective warp)
  - **euclidean**: Rotation + translation only (most restrictive, good for cameras that are well-aligned)

**Advanced options:**

- `--no-temporal`: Disable temporal alignment for main videos (temporal_shift will be 0)
- `--main-only` / `--calibration-only`: Process only main or calibration videos
- `--rejected-only`: Reprocess only samples marked as rejected in step1c review
- `--preview-only`: Regenerate preview images from existing alignments (no re-alignment)
- `--use-initial-transform SAMPLE_ID`: Use alignment from SAMPLE_ID as starting point for optimization
- `--save-as-template`: Save alignment as 'initial_transform' in config for reuse
- `--force`: Reprocess even if alignment already exists

**Initial transform workflow:**

You can manually add an initial transform to `config.json` to provide a starting point for alignment:

```json
{
  "alignment_main": {
    "initial_transform": [
      [0.998, -0.052, 45.2],
      [0.051, 0.997, -12.3],
      [0, 0, 1]
    ]
  }
}
```

Or use a known-good sample as template:

```bash
# Process one sample and save as template
python scripts/step1b_run_alignments.py --samples 001 --save-preview --save-as-template

# Use that template for all other samples
python scripts/step1b_run_alignments.py --all --save-preview --use-initial-transform 001
```

**Output:**

- Alignment parameters in `config.json`
- 3 preview images per video pair (if `--save-preview` used):
  - `{id}_main_vis.png` - VIS reference
  - `{id}_main_uv_aligned.png` - Aligned UV
  - `{id}_main_composite.png` - 50% overlay for checking alignment
  - Same for calibration videos: `{id}_calibration_*.png`

**Recommended workflow:**

1. Run with `--save-preview` to calculate alignment and save previews
2. Review composites in step1c
3. If some alignments rejected, reprocess with `--rejected-only` and different settings:
   ```bash
   python scripts/step1b_run_alignments.py --all --rejected-only --save-preview --main-motion-type euclidean
   ```
4. Repeat step1c review until all alignments approved

---

### Step 1c: Review Alignments (Manual - 2-3 min)

```bash
# Review all samples with composites
python scripts/step1c_review_alignments.py --all

# Review specific samples
python scripts/step1c_review_alignments.py --samples 001 006 012

# Review only main videos (skip calibration)
python scripts/step1c_review_alignments.py --all --main-only
python scripts/step1c_review_alignments.py --all --skip-calibration  # same as --main-only
```

**What it does:**

- Shows composite images full-screen
- Press Y=Approve, N=Reject, S=Skip
- Marks each alignment in `config.json`
- Auto-advances to next sample

**Interactive controls:**

- **Y or SPACE**: Approve this alignment
- **N or R**: Reject this alignment (will need to reprocess)
- **S**: Skip (leave unmarked)
- **Q or ESC**: Quit review

**What to check:**

- UV and VIS should overlap perfectly
- No ghosting or color fringing
- Features should line up (edges, landmarks, etc.)

**Output:** `review_status: "approved"` or `"rejected"` in `config.json`

---

### Step 2: Extract Calibration Patches (Manual/Auto - 15 min)

```bash
# STANDARD WORKFLOW: Auto-detect ArUco, show for verification
python scripts/step2_extract_calibration.py --all

# Auto-only: skip manual if ArUco fails
python scripts/step2_extract_calibration.py --all --auto-only

# Specific samples
python scripts/step2_extract_calibration.py --samples 001 006 012

# Specify different patch count (default: 28 from autolinearizer)
python scripts/step2_extract_calibration.py --all --num-patches 24
python scripts/step2_extract_calibration.py --samples 005 014 --num-patches 8

# Use different autolinearizer
python scripts/step2_extract_calibration.py --all --autolinearizer data/autolinearizer_custom.json

# Extract from different frame (not first frame)
python scripts/step2_extract_calibration.py --all --frame-offset 10

# Force re-extraction
python scripts/step2_extract_calibration.py --samples 001 --force

# Generate analysis plots/metrics (for already-processed samples)
python scripts/step2_extract_calibration.py --all --export-analysis

# Use different calibration/camera CSVs
python scripts/step2_extract_calibration.py --all \
  --calibration-csv data/aruco_samples.csv \
  --camera-csv data/camera_sensitivities.csv
```

**What it does:**

- Loads first frame of calibration videos (or specified `--frame-offset`)
- Applies saved alignment
- Auto-detects ArUco markers → estimates patch positions using autolinearizer
- Shows for verification (can edit/accept/reject)
- Falls back to manual clicking if ArUco fails
- Extracts pixel values from patches
- Generates quality analysis plots (linearization curves, R² metrics)
- Saves to `videos/samples/{id}/config.json` and `videos/samples/{id}/analysis/`

**Interactive controls (auto-detected mode):**

- **Y or SPACE**: Accept detected patches
- **E**: Edit patch positions manually
- **R**: Retry detection
- **N**: Change patch count
- **S**: Skip this sample
- **Q**: Quit

**Interactive controls (manual mode):**

- **Left Click**: Add patch
- **Right Click**: Remove last patch
- **SPACE**: Done (accept patches)
- **S**: Skip
- **Q**: Quit
- **NOTE:** This can be a little slow to respond to mouse clicks so you have to be deliberate when manually adding points. Click the mouse but don't move it until you see the point show up. If you are moving very quickly (easy to do during batch processing), you will see new points show up a little after where you click. Slow down and it works just fine.

**Options:**

- `--auto-only`: Skip manual intervention if ArUco detection fails
- `--num-patches N`: Number of patches to extract (default: auto from autolinearizer)
- `--autolinearizer PATH`: Path to autolinearizer JSON (default: `data/autolinearizer_custom.json`)
- `--frame-offset N`: Frame number to extract (default: 0 = first frame)
- `--export-analysis`: Generate analysis plots for already-processed samples
- `--calibration-csv PATH`: Path to calibration reflectance CSV
- `--camera-csv PATH`: Path to camera sensitivities CSV
- `--force`: Re-extract even if patches already exist

**Output:**

- `calibration_patches` in `config.json` with:
  - Patch positions
  - Pixel values (VIS and UV)
  - ArUco corners (if detected)
  - Number of patches
- Analysis plots in `videos/samples/{id}/analysis/`:
  - `linearization_quality.png`: Camera response curve, linearization curve, accuracy
  - `calibration_metrics.json`: MAE and R² per channel
  - `calibration_metrics.txt`: Human-readable metrics

---

### Step 2b (Optional): Create Custom Autolinearizer

If the default autolinearizer doesn't position patches correctly, create your own:

```bash
# Create custom autolinearizer from sample with good ArUco markers
python scripts/create_autolinearizer.py --sample 006

# This creates data/autolinearizer_custom.json
# Replace the default:
cp data/autolinearizer.json data/autolinearizer_original.json
cp data/autolinearizer_custom.json data/autolinearizer.json

# Or use via command line:
python scripts/step2_extract_calibration.py --all --autolinearizer data/autolinearizer_custom.json
```

**What it does:**

- Detects 4 ArUco markers in your calibration frame
- You manually click on the center of each color patch (28 patches)
- Saves positions relative to the detected markers
- All future samples use homography transformation based on YOUR reference

**Interactive controls:**

- **Left click**: Add patch
- **Right click**: Remove last patch
- **R**: Reset all patches
- **SPACE**: Save when done
- **Q**: Quit without saving

---

### Step 3: Apply Full Pipeline (Automated - Overnight)

**BEFORE RUNNING:** Edit `videos/samples/pipeline_config.json` with your settings and save in a new locations:

```json
{
  "camera_sensitivities_path": "data/calibrations/cameras/a7siii_300to700_4sensor.csv",
  "calibration_values_path": {
    "8": "data/calibrations/standards/microarray_trunc_25to50.csv",
    "28": "data/calibrations/standards/aruco_samples_black_card.csv"
  },
  "is_sony_camera": true,
  "animal_type": "apis",
  "output_format": "mp4",
  "batch_size": 32
}
```

**Notes**

- `is_sony_camera` - if `true` then it will force the algorithm to use the SLog3 curve. If you are using a "truncated" color standard, then you likely want to set this to false.
- `calibration_values_path`: This object maps keys as the patch count to a specific calibration file. If you have two different sets of calibration standards with the same number of patches, you will have to separate them into different batches.

**Available animal types:**

- `apis` - Honeybee (Apis mellifera)
- `avian` - Generic bird vision
- `bluetit` - Blue tit
- `bombus_terrestris_dalmaticus` - Bumblebee subspecies
- `bombus_terrestris_sassaricus` - Bumblebee subspecies
- `domestic_chick` - Domestic chick
- `jumping_spider` - Jumping spider
- `dog` - Dog vision
- `mouse` - Mouse vision

```bash
# STANDARD WORKFLOW: Process only approved samples (recommended)
python scripts/step3_apply_full_pipeline.py --approved-only

# Override animal type from command line
python scripts/step3_apply_full_pipeline.py --approved-only --animal avian

# Process specific samples
python scripts/step3_apply_full_pipeline.py --samples 001 005 --animal apis

# Process all samples (ignoring review status)
python scripts/step3_apply_full_pipeline.py --all --animal bombus_terrestris_dalmaticus

# Generate analysis plots and visualizations
python scripts/step3_apply_full_pipeline.py --approved-only --save-analysis

# Preview mode: process only first 30 frames
python scripts/step3_apply_full_pipeline.py --samples 001 --preview
python scripts/step3_apply_full_pipeline.py --samples 001 --preview 100  # first 100 frames

# Use different output directory
python scripts/step3_apply_full_pipeline.py --approved-only --output-dir videos/processed

# Adjust batch size (if running out of memory)
python scripts/step3_apply_full_pipeline.py --approved-only --batch-size 16

# Force reprocessing
python scripts/step3_apply_full_pipeline.py --samples 001 --force

# Use custom pipeline config
python scripts/step3_apply_full_pipeline.py --approved-only --pipeline-config /path/to/config.json
```

**What it does:**

- Loads alignment parameters from `config.json`
- Builds linearizer from calibration patch data
- Loads sense converter for animal vision
- Applies complete pipeline in single pass:
  - Loader → Flip → Warp (alignment) → Linearizer → SenseConverter → Writer
- No intermediate video exports (avoids recompression)
- Outputs TWO videos per sample:
  - Animal vision (3-4 channels depending on animal)
  - Human vision (RGB from linearized VIS channels)

**Options:**

- `--approved-only`: Only process samples with approved alignments (recommended)
- `--all`: Process all samples (ignore review status)
- `--samples ID1 ID2 ...`: Process specific samples
- `--animal TYPE`: Override animal type from pipeline config
- `--pipeline-config PATH`: Custom pipeline config JSON
- `--output-dir DIR`: Output directory (default: `videos/output`)
- `--batch-size N`: Frames per batch (default: 32)
- `--save-analysis`: Generate analysis plots and visualizations
- `--preview [N]`: Preview mode - process only first N frames (default: 30)
- `--force`: Reprocess even if output already exists
- `--samples-dir DIR`: Custom samples directory

**Pipeline components:**

1. **Linearization**: Removes camera post-processing using calibration patches
   - Sony SLog3 PowerLaw (if `is_sony_camera: true`)
   - Polynomial power law fitting (otherwise)
2. **Animal Vision Conversion**: Transforms linearized camera values to animal photoreceptor responses
   - Uses pre-built sense converters (linear transformation matrix)
   - Based on FReD flower database and animal spectral sensitivities

**Output:**

```
videos/output/001/
├── 001_animal_apis.mp4      # Animal vision video
└── 001_human.mp4             # Human-visible representation
```

If `--save-analysis` is used:

```
videos/output/001/analysis/
├── alignment_ghosting.png          # R/B from VIS, G from UV (shows misalignment)
├── linearization_quality.png       # Camera response, linearization curve, accuracy
├── linearization_metrics.json      # MAE and R² per channel
├── sensitivity_comparison.png      # Camera vs animal spectral sensitivities
├── conversion_accuracy.png         # Predicted vs actual animal responses
└── conversion_metrics.json         # MAE and R² per photoreceptor
```

**Preview mode:**

Use `--preview` to quickly test the pipeline on a small number of frames:

```bash
# Process first 30 frames (default)
python scripts/step3_apply_full_pipeline.py --samples 001 --preview

# Process first 100 frames
python scripts/step3_apply_full_pipeline.py --samples 001 --preview 100
```

Preview outputs have `_preview` suffix: `001_animal_apis_preview.mp4`, `001_human_preview.mp4`

**Performance:** ~11 minutes per sample (depends on video length and hardware)

---

## File Structure

```
videos/samples/001/
├── VIS_VID*.MP4                        # Main visible video
├── UV_VIS*.MP4                         # Main UV video
├── calibration/
│   ├── VIS_VID*.MP4                    # Calibration with color standard
│   └── UV_VIS*.MP4
├── config.json                         # All settings and parameters
├── 001_main_vis.png                    # Step 1b preview
├── 001_main_uv_aligned.png
├── 001_main_composite.png
├── 001_calibration_vis.png
├── 001_calibration_uv_aligned.png
├── 001_calibration_composite.png
└── analysis/                           # Step 2 analysis (optional)
    ├── linearization_quality.png
    ├── calibration_metrics.json
    └── calibration_metrics.txt
```

**config.json structure:**

```json
{
  "sample_id": "001",
  "flip_main": "vertical",
  "flip_calibration": "vertical",
  "has_calibration": true,
  "alignment_main": {
    "method": "ecc",
    "motion_type": "homography",
    "flip": "vertical",
    "homography_matrix": [[...], [...], [...]],
    "temporal_shift": 0,
    "output_size": [1920, 1080],
    "review_status": "approved",
    "has_homography": true,
    "processing_time": 85.3,
    "vis_frame_count": 3600,
    "uv_frame_count": 3600,
    "usable_frame_count": 3600
  },
  "alignment_calibration": {
    "method": "aruco",
    "motion_type": "homography",
    "flip": "vertical",
    "homography_matrix": [[...], [...], [...]],
    "temporal_shift": 0,
    "output_size": [1920, 1080],
    "review_status": "approved",
    "has_homography": true,
    "processing_time": 28.1
  },
  "calibration_patches": {
    "aruco_detected": true,
    "num_patches": 28,
    "frame_offset": 0,
    "patch_positions": [{"x": 120, "y": 80}, ...],
    "patch_values_vis": [{"r": 0.45, "g": 0.52, "b": 0.38}, ...],
    "patch_values_uv": [{"b": 0.35, "g": 0.38, "r": 0.41}, ...],
    "aruco_corners": [[...], [...], [...], [...]]
  }
}
```

---

## Time Breakdown for 40 Samples

**Your time (manual):**

- Step 1a: 10 minutes (flip selection)
- Step 1c: 2-3 minutes (review composites)
- Step 2: 15 minutes (calibration patches, mostly auto)
- **Total: ~30 minutes**

**Computer time (automated):**

- Step 1b: ~1-2 hours (ECC main + ArUco calibration)
- Step 3: Many hours (full video processing, ~11 min/sample)
- **Runs unattended**

---

## Troubleshooting

### Step 1b: "AlignmentNotFound" error

- Check composite preview - is alignment actually bad?
- Try different motion model:
  ```bash
  python scripts/step1b_run_alignments.py --samples 001 --save-preview --main-motion-type euclidean --force
  ```
- Verify flip selection in step1a was correct
- Try using initial transform from a known-good sample:
  ```bash
  python scripts/step1b_run_alignments.py --samples 001 --save-preview --use-initial-transform 006 --force
  ```

### Step 1c: Composite looks misaligned

- Reject it (press N)
- Re-run step1b with different settings:
  ```bash
  python scripts/step1b_run_alignments.py --samples 001 --save-preview --main-motion-type euclidean --force
  ```
- Or manually edit `config.json` to adjust parameters

### Step 2: ArUco detection fails

- Script falls back to manual mode automatically
- Click patch centers manually
- Or skip with `--auto-only` flag
- Consider creating custom autolinearizer from a sample with good markers

### Step 3: Video size mismatch error

- Run step1b again to regenerate alignment with correct `output_size`
- The alignment output size must match VIS video dimensions

### Temporal shift issues

- Manually edit `temporal_shift` in `config.json` if needed
- Re-run step3 with `--force`
- Or disable temporal alignment in step1b: `--no-temporal`

---

## Tips

1. **Batch manual work** - Do all clicking at once, then walk away
2. **Review samples strategically** - Check a few, not all 40
3. **Use --auto-only for step2** - Skip manual intervention for calibration
4. **No intermediate exports** - Videos stay in memory until final output
5. **Config.json is your friend** - All parameters saved, can manually edit
6. **Use preview mode** - Test step3 quickly with `--preview` before full processing
7. **Save analysis** - Use `--save-analysis` to generate quality metrics

---

## Examples

### Single sample end-to-end:

```bash
python scripts/step1a_select_flips.py --samples 001
python scripts/step1b_run_alignments.py --samples 001 --save-preview
python scripts/step1c_review_alignments.py --samples 001
python scripts/step2_extract_calibration.py --samples 001
python scripts/step3_apply_full_pipeline.py --samples 001
```

### Batch all 40 samples (recommended):

```bash
# Morning: Manual work (~30 min)
python scripts/step1a_select_flips.py --all
python scripts/step1b_run_alignments.py --all --save-preview  # Wait ~1-2 hrs
python scripts/step1c_review_alignments.py --all
python scripts/step2_extract_calibration.py --all

# Evening: Start overnight processing
python scripts/step3_apply_full_pipeline.py --approved-only --animal apis
```

### Using a different samples directory:

```bash
# All scripts support --samples-dir
export SAMPLES_DIR="/path/to/other/videos"

python scripts/step1a_select_flips.py --all --samples-dir $SAMPLES_DIR
python scripts/step1b_run_alignments.py --all --save-preview --samples-dir $SAMPLES_DIR
python scripts/step1c_review_alignments.py --all --samples-dir $SAMPLES_DIR
python scripts/step2_extract_calibration.py --all --samples-dir $SAMPLES_DIR
python scripts/step3_apply_full_pipeline.py --approved-only --samples-dir $SAMPLES_DIR
```

### Troubleshoot a single problematic sample:

```bash
# Force re-align with different method
python scripts/step1b_run_alignments.py --samples 005 --main-method any --save-preview --force

# Try euclidean motion model
python scripts/step1b_run_alignments.py --samples 005 --main-motion-type euclidean --save-preview --force

# Manual calibration patches
python scripts/step2_extract_calibration.py --samples 005 --force

# Test with preview mode first
python scripts/step3_apply_full_pipeline.py --samples 005 --preview

# Process just this one
python scripts/step3_apply_full_pipeline.py --samples 005 --force
```

### Reprocess rejected alignments:

```bash
# After step1c review, reprocess all rejected samples with euclidean motion
python scripts/step1b_run_alignments.py --all --rejected-only --save-preview --main-motion-type euclidean

# Review again
python scripts/step1c_review_alignments.py --all

# Repeat until all approved
```

---

## What You Get After Step 3

For each processed sample, you'll have two output videos in `videos/output/{sample_id}/`:

1. **`{sample_id}_animal_{animal_type}.mp4`** - Animal vision video

   - What the animal actually sees based on their photoreceptor responses
   - For honeybees (apis): UV, Blue, Green vision combined
   - For birds (avian): Tetrachromatic vision (4 channels)

2. **`{sample_id}_human.mp4`** - Human-visible reference video
   - RGB representation from the linearized visible channels
   - Useful for comparison and verification

**Example output structure:**

```
videos/output/
├── 001/
│   ├── 001_animal_apis.mp4      # Honeybee vision
│   ├── 001_human.mp4             # Human reference
│   └── analysis/                 # If --save-analysis used
│       ├── alignment_ghosting.png
│       ├── linearization_quality.png
│       ├── linearization_metrics.json
│       ├── sensitivity_comparison.png
│       ├── conversion_accuracy.png
│       └── conversion_metrics.json
├── 006/
│   ├── 006_animal_apis.mp4
│   └── 006_human.mp4
└── ...
```
