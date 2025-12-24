#!/bin/bash

# Get the directory where the script is located (ReSQLver directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
# Input: The output of the tree generation step
# Note: This points to the parent directory. compare.py will recursively search for JSON files.
TREE_DATASET_DIR="$PROJECT_ROOT/ReSQLver/outputs/train_tree_dataset_full"

# Intermediate and Final Outputs
COMPARE_OUTPUT_DIR="$PROJECT_ROOT/ReSQLver/outputs/compare_dataset"
DPO_OUTPUT_DIR="$PROJECT_ROOT/ReSQLver/outputs/final_dataset"
CLEAN_OUTPUT_DIR="$PROJECT_ROOT/ReSQLver/outputs/final_clear_dataset"
FINAL_MERGED_FILE="$PROJECT_ROOT/ReSQLver/outputs/final_data_emerge.json"

# Python Scripts
COMPARE_SCRIPT="$SCRIPT_DIR/compare/compare.py"
EXTRACT_SCRIPT="$SCRIPT_DIR/compare/final_data_generate.py"
CLEAN_SCRIPT="$SCRIPT_DIR/compare/delete_equal_sql.py"
MERGE_SCRIPT="$SCRIPT_DIR/compare/date_emerge.py"

echo "=================================================="
echo "Starting Comparison Pipeline"
echo "=================================================="
echo "Tree Dataset Dir: $TREE_DATASET_DIR"
echo "Final Output File: $FINAL_MERGED_FILE"
echo "=================================================="

# Step 1: Generate Comparison Trees
echo ""
echo "[Step 1] Generating Comparison Trees..."
python "$COMPARE_SCRIPT" \
    --input_dir "$TREE_DATASET_DIR" \
    --output_dir "$COMPARE_OUTPUT_DIR"

# Step 2: Extract DPO Data
echo ""
echo "[Step 2] Extracting DPO Data..."
python "$EXTRACT_SCRIPT" \
    --input_dir "$COMPARE_OUTPUT_DIR" \
    --output_dir "$DPO_OUTPUT_DIR"

# Step 3: Clean DPO Data (Remove equal pairs)
echo ""
echo "[Step 3] Cleaning DPO Data..."
python "$CLEAN_SCRIPT" \
    --input_dir "$DPO_OUTPUT_DIR" \
    --output_dir "$CLEAN_OUTPUT_DIR"

# Step 4: Merge into Final JSON
echo ""
echo "[Step 4] Merging into Final JSON..."
python "$MERGE_SCRIPT" \
    --input_dir "$CLEAN_OUTPUT_DIR" \
    --output_file "$FINAL_MERGED_FILE"

echo ""
echo "=================================================="
echo "Pipeline Complete!"
echo "Final Output: $FINAL_MERGED_FILE"
echo "=================================================="
