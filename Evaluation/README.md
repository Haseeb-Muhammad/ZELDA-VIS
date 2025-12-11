# Cell Tracking Challenge Evaluation Scripts

This directory contains scripts for evaluating cell segmentation predictions using the Cell Tracking Challenge (CTC) evaluation software.

**✨ Works with both 2D and 3D TIFF images** - The SEGMeasure evaluation software natively supports both 2D and 3D datasets without any special configuration.

## Files

- **`run_evaluation.sh`** - Main bash script that orchestrates the evaluation process
- **`prepare_and_run_evaluation.py`** - Python script that handles file restructuring and evaluation execution

## Requirements

- Python 3
- Python packages:
  - `tifffile` (for image dimension detection)
  - `numpy` (for numerical operations)
  - `scipy` (for optimal instance matching)
  - `scikit-image` (for image I/O)
  - `numba` (for JIT compilation - performance optimization)
  - `tqdm` (for progress bars)
- Cell Tracking Challenge SEGMeasure executable
- Ground truth and prediction TIFF images (2D or 3D)

### Installing Python Dependencies

```bash
pip install tifffile numpy scipy scikit-image numba tqdm
```

## Usage

### Basic Usage

```bash
./run_evaluation.sh <gt_folder> <predictions_folder> <eval_software_path> <dataset_name>
```

### Arguments

- `gt_folder` - Path to folder containing ground truth images (TIFF format)
- `predictions_folder` - Path to folder containing prediction images (TIFF format)
- `eval_software_path` - Path to SEGMeasure executable or its parent directory
- `dataset_name` - Name of the dataset (used for organizing results)

### Example

```bash
./run_evaluation.sh \
    /netscratch/muhammad/ProcessedDatasets/Fluo-N3DH-CHO/test/masks \
    /netscratch/muhammad/predictions/CHO \
    /netscratch/muhammad/codes/CellSegmentation/evalSoftware/Linux \
    Fluo-N3DH-CHO
```

## What It Does

1. **Validates inputs** - Checks that all paths exist and contain TIFF files
2. **Detects dimensionality** - Automatically determines if images are 2D or 3D by analyzing TIFF structure
3. **Restructures files** - Copies and renames files according to CTC naming convention:
   - Ground truth: `man_seg0000.tif`, `man_seg0001.tif`, ...
   - Predictions: `mask0000.tif`, `mask0001.tif`, ...
4. **Runs SEGMeasure evaluation** - Executes the CTC SEGMeasure tool (works with both 2D and 3D)
5. **Calculates mAP metrics** - Computes mean Average Precision (mAP) at multiple IoU thresholds
6. **Saves results** - Stores mapping, SEGMeasure results, and mAP metrics in organized directory structure
7. **Cleans up** - Automatically removes temporary files

## Output

Results are saved to: `<script_directory>/results/<dataset_name>/`

Output files:
- `mapping.json` - Maps original filenames to CTC convention names
- `evaluation_results.txt` - Combined SEGMeasure and mAP metrics with per-image details
- `map_results.json` - Detailed mAP results in JSON format including per-image breakdowns
- Additional result files (CSV, log files) from the evaluation software (if any)

### Metrics Calculated

**SEGMeasure (CTC Official Metric):**
- SEG measure score (0-1 scale, higher is better)

**mAP (mean Average Precision):**
- Overall mAP across IoU thresholds 0.5-0.95 (COCO-style)
- AP@0.50 (Average Precision at IoU threshold 0.5)
- AP@0.75 (Average Precision at IoU threshold 0.75)
- Per-image mAP scores
- Instance counts (predicted vs ground truth)

## File Naming Convention

The script automatically matches prediction files to ground truth files using:
1. Exact stem match (e.g., `image001.tif` matches `image001.tif`)
2. Substring match (e.g., `image001_masks.tif` matches `image001.tif`)
3. Numeric frame match (extracts numbers from filenames)

## Advanced Options

You can also call the Python script directly for more control:

```bash
python3 prepare_and_run_evaluation.py \
    --gt /path/to/gt \
    --pred /path/to/predictions \
    --eval /path/to/evalSoftware \
    --dataset my_dataset \
    --output-dir /custom/output/path \
    --name TMP \
    --seq 01 \
    --digits 4 \
    --keep-temp
```

Options:
- `--output-dir` - Custom output directory for results (default: `results/<dataset>` in script directory)
- `--name` - Base name for temporary dataset folder (default: TMP)
- `--seq` - Sequence ID (default: 01)
- `--digits` - Zero-padding for filenames (default: 4)
- `--keep-temp` - Preserve temporary folder for debugging

## 2D vs 3D Image Support

The Cell Tracking Challenge SEGMeasure tool **natively handles both 2D and 3D** images automatically:

- **2D Images**: Single-slice TIFF files (evaluated as 2D segmentation)
- **3D Images**: Multi-slice TIFF stacks (evaluated as 3D segmentation)

No special configuration is needed - SEGMeasure detects the dimensionality from the TIFF file structure and processes accordingly. The script will display detected dimensionality for informational purposes:
```
[INFO] Detected image dimensionality: 2D
[INFO] SEGMeasure will automatically process 2D images
```

This evaluation approach is consistent with the Cell Tracking Challenge benchmarks which include both 2D datasets (like many microscopy images) and 3D datasets (like volumetric/stack data).

## Notes

- For each prediction file, there should be a corresponding ground truth file
- Files must be in TIFF format (`.tif` or `.tiff`)
- Works with both 2D and 3D TIFF images (automatically detected)
- The script handles automatic file matching even with different naming patterns
- Temporary files are automatically cleaned up unless `--keep-temp` is specified
- Make sure `tifffile` Python package is installed: `pip install tifffile`
