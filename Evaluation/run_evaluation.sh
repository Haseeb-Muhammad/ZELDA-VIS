#!/bin/bash

###############################################################################
# Cell Tracking Challenge Evaluation Wrapper Script
#
# This script orchestrates the evaluation of cell segmentation predictions
# against ground truth images using the Cell Tracking Challenge evaluation
# software.
#
# Usage:
#   ./run_evaluation.sh <gt_folder> <predictions_folder> <eval_software_path> <dataset_name>
#
# Arguments:
#   gt_folder            - Path to folder containing ground truth images
#   predictions_folder   - Path to folder containing prediction images  
#   eval_software_path   - Path to SEGMeasure executable or its parent directory
#   dataset_name         - Name of the dataset (for organizing results)
#
# The script will:
#   1. Validate input arguments and paths
#   2. Run Python script to restructure files according to CTC naming convention
#   3. Execute the SEGMeasure evaluation tool
#   4. Calculate mean Average Precision (mAP) metrics
#   5. Display detailed evaluation results
#   6. Clean up temporary files automatically
#   7. Save naming mapping and results to output files
#
# Output files (saved in the same directory as this script):
#   - mapping.json                         - Original to CTC name mapping
#   - evaluation_results.txt               - SEGMeasure and mAP metrics
#   - map_results.json                     - Detailed mAP results per image
#
###############################################################################

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    echo "Usage: $0 <gt_folder> <predictions_folder> <eval_software_path> <dataset_name>"
    echo ""
    echo "Arguments:"
    echo "  gt_folder            Path to folder containing ground truth images"
    echo "  predictions_folder   Path to folder containing prediction images"
    echo "  eval_software_path   Path to SEGMeasure executable or its parent directory"
    echo "  dataset_name         Name of the dataset (for organizing results)"
    echo ""
    exit 1
}

# Check if correct number of arguments provided
if [ "$#" -ne 4 ]; then
    print_error "Incorrect number of arguments. Expected 4, got $#"
    usage
fi

# Parse arguments
GT_FOLDER="$1"
PRED_FOLDER="$2"
EVAL_SOFTWARE="$3"
DATASET_NAME="$4"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/prepare_and_run_evaluation.py"

# Validate that input directories exist
print_info "Validating input paths..."

if [ ! -d "$GT_FOLDER" ]; then
    print_error "Ground truth folder does not exist: $GT_FOLDER"
    exit 1
fi

if [ ! -d "$PRED_FOLDER" ]; then
    print_error "Predictions folder does not exist: $PRED_FOLDER"
    exit 1
fi

if [ ! -e "$EVAL_SOFTWARE" ]; then
    print_error "Evaluation software path does not exist: $EVAL_SOFTWARE"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Python script not found at: $PYTHON_SCRIPT"
    print_error "Make sure prepare_and_run_evaluation.py is in the same directory as this script"
    exit 1
fi

print_success "All input paths validated"

# Count files in directories
GT_COUNT=$(find "$GT_FOLDER" -maxdepth 1 -type f \( -iname "*.tif" -o -iname "*.tiff" \) | wc -l)
PRED_COUNT=$(find "$PRED_FOLDER" -maxdepth 1 -type f \( -iname "*.tif" -o -iname "*.tiff" \) | wc -l)

print_info "Ground truth images found: $GT_COUNT"
print_info "Prediction images found: $PRED_COUNT"

if [ "$GT_COUNT" -eq 0 ]; then
    print_error "No TIFF images found in ground truth folder"
    exit 1
fi

if [ "$PRED_COUNT" -eq 0 ]; then
    print_error "No TIFF images found in predictions folder"
    exit 1
fi

# Determine results directory (in the same directory as this script)
RESULTS_DIR="${SCRIPT_DIR}/results/${DATASET_NAME}"

print_info "Starting evaluation process..."
echo ""
print_info "Configuration:"
echo "  Ground Truth:  $GT_FOLDER"
echo "  Predictions:   $PRED_FOLDER"
echo "  Eval Software: $EVAL_SOFTWARE"
echo "  Dataset Name:  $DATASET_NAME"
echo "  Results Dir:   $RESULTS_DIR"
echo ""

# Run the Python evaluation script
print_info "Running Python evaluation script..."
echo "=================================================="

# Capture output and run Python script
# The --keep-temp flag is NOT used, so temporary files will be auto-cleaned
TEMP_OUTPUT=$(mktemp)

if python3 "$PYTHON_SCRIPT" \
    --gt "$GT_FOLDER" \
    --pred "$PRED_FOLDER" \
    --eval "$EVAL_SOFTWARE" \
    --dataset "$DATASET_NAME" \
    --output-dir "$RESULTS_DIR" \
    --name "EVAL" \
    --seq "01" \
    --digits 4 | tee "$TEMP_OUTPUT"; then
    
    print_success "Evaluation completed successfully"
    print_success "Results saved to: $RESULTS_DIR"
    
else
    EVAL_EXIT_CODE=$?
    print_error "Evaluation failed with exit code: $EVAL_EXIT_CODE"
    rm -f "$TEMP_OUTPUT"
    exit $EVAL_EXIT_CODE
fi

# Clean up temporary output file
rm -f "$TEMP_OUTPUT"

echo ""
echo "=================================================="
print_success "Evaluation process completed"
echo ""
print_info "Output files:"
echo "  - Mapping: ${RESULTS_DIR}/mapping.json"
echo "  - Results: ${RESULTS_DIR}/evaluation_results.txt"
echo "  - mAP Results: ${RESULTS_DIR}/map_results.json"
echo ""

# Display summary of results if available
RESULTS_FILE="${RESULTS_DIR}/evaluation_results.txt"
if [ -f "$RESULTS_FILE" ]; then
    print_info "Evaluation Results Summary:"
    echo "=================================================="
    cat "$RESULTS_FILE"
    echo "=================================================="
fi

exit 0
