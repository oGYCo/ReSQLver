#!/bin/bash

# Get the directory where the script is located (ReSQLver directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# The workspace root (parent of ReSQLver)
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

# ==========================================================
# Configuration
# ==========================================================

# Model and Data Paths
MODEL_PATH="$WORKSPACE_ROOT/Qwen2.5-Coder-3B-Instruct"
INPUT_FILE="$WORKSPACE_ROOT/data/train_bird.json"
DEV_FILE="$WORKSPACE_ROOT/data/train/train.json"
DB_PATH="$WORKSPACE_ROOT/data/train/train_databases"
GOLD_FILE="$WORKSPACE_ROOT/data/train/train_gold.sql"

# Output Directories
TREE_OUTPUT_BASE="$WORKSPACE_ROOT/ReSQLver/outputs/train_tree_dataset_full"
COMPARE_OUTPUT_DIR="$WORKSPACE_ROOT/ReSQLver/outputs/compare_dataset"
DPO_OUTPUT_DIR="$WORKSPACE_ROOT/ReSQLver/outputs/final_dataset"
CLEAN_OUTPUT_DIR="$WORKSPACE_ROOT/ReSQLver/outputs/final_clear_dataset"
FINAL_MERGED_FILE="$WORKSPACE_ROOT/ReSQLver/outputs/final_data_emerge.json"

# Python Scripts
BUILD_TREE_SCRIPT="$SCRIPT_DIR/tree/build_tree_dataset.py"
COMPARE_SCRIPT="$SCRIPT_DIR/compare/compare.py"
EXTRACT_SCRIPT="$SCRIPT_DIR/compare/final_data_generate.py"
CLEAN_SCRIPT="$SCRIPT_DIR/compare/delete_equal_sql.py"
MERGE_SCRIPT="$SCRIPT_DIR/compare/date_emerge.py"

# Create logs directory
mkdir -p "$WORKSPACE_ROOT/logs"
mkdir -p "$TREE_OUTPUT_BASE"

# ==========================================================
# Part 1: Parallel Tree Generation
# ==========================================================

echo "=================================================="
echo "Step 1: Starting Parallel Tree Generation (8 GPUs)"
echo "=================================================="

# GPU 0 (Items 0-1179)
echo "Launching GPU 0 job (0-1179)..."
nohup python "$BUILD_TREE_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --dev_file "$DEV_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${TREE_OUTPUT_BASE}/part0" \
    --start_idx 0 --end_idx 1179 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 0 \
    --early_stop \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/train_gpu0.log" 2>&1 &
PID0=$!

# GPU 1 (Items 1179-2357)
echo "Launching GPU 1 job (1179-2357)..."
nohup python "$BUILD_TREE_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --dev_file "$DEV_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${TREE_OUTPUT_BASE}/part1" \
    --start_idx 1179 --end_idx 2357 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 1 \
    --early_stop \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/train_gpu1.log" 2>&1 &
PID1=$!

# GPU 2 (Items 2357-3536)
echo "Launching GPU 2 job (2357-3536)..."
nohup python "$BUILD_TREE_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --dev_file "$DEV_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${TREE_OUTPUT_BASE}/part2" \
    --start_idx 2357 --end_idx 3536 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 2 \
    --early_stop \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/train_gpu2.log" 2>&1 &
PID2=$!

# GPU 3 (Items 3536-4714)
echo "Launching GPU 3 job (3536-4714)..."
nohup python "$BUILD_TREE_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --dev_file "$DEV_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${TREE_OUTPUT_BASE}/part3" \
    --start_idx 3536 --end_idx 4714 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 3 \
    --early_stop \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/train_gpu3.log" 2>&1 &
PID3=$!

# GPU 4 (Items 4714-5893)
echo "Launching GPU 4 job (4714-5893)..."
nohup python "$BUILD_TREE_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --dev_file "$DEV_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${TREE_OUTPUT_BASE}/part4" \
    --start_idx 4714 --end_idx 5893 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 4 \
    --early_stop \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/train_gpu4.log" 2>&1 &
PID4=$!

# GPU 5 (Items 5893-7071)
echo "Launching GPU 5 job (5893-7071)..."
nohup python "$BUILD_TREE_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --dev_file "$DEV_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${TREE_OUTPUT_BASE}/part5" \
    --start_idx 5893 --end_idx 7071 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 5 \
    --early_stop \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/train_gpu5.log" 2>&1 &
PID5=$!

# GPU 6 (Items 7071-8250)
echo "Launching GPU 6 job (7071-8250)..."
nohup python "$BUILD_TREE_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --dev_file "$DEV_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${TREE_OUTPUT_BASE}/part6" \
    --start_idx 7071 --end_idx 8250 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 6 \
    --early_stop \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/train_gpu6.log" 2>&1 &
PID6=$!

# GPU 7 (Items 8250-9428)
echo "Launching GPU 7 job (8250-9428)..."
nohup python "$BUILD_TREE_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --dev_file "$DEV_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${TREE_OUTPUT_BASE}/part7" \
    --start_idx 8250 --end_idx 9428 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 7 \
    --early_stop \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/train_gpu7.log" 2>&1 &
PID7=$!

echo "All jobs launched. PIDs: $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7"
echo "Waiting for completion... (This may take a while)"

# Wait for all background processes to finish
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7

echo "Tree generation complete!"

# ==========================================================
# Part 2: Comparison Pipeline
# ==========================================================

echo ""
echo "=================================================="
echo "Step 2: Starting Comparison Pipeline"
echo "=================================================="

# Step 2.1: Generate Comparison Trees
echo "[2.1] Generating Comparison Trees..."
python "$COMPARE_SCRIPT" \
    --input_dir "$TREE_OUTPUT_BASE" \
    --output_dir "$COMPARE_OUTPUT_DIR"

# Step 2.2: Extract DPO Data
echo ""
echo "[2.2] Extracting DPO Data..."
python "$EXTRACT_SCRIPT" \
    --input_dir "$COMPARE_OUTPUT_DIR" \
    --output_dir "$DPO_OUTPUT_DIR"

# Step 2.3: Clean DPO Data
echo ""
echo "[2.3] Cleaning DPO Data..."
python "$CLEAN_SCRIPT" \
    --input_dir "$DPO_OUTPUT_DIR" \
    --output_dir "$CLEAN_OUTPUT_DIR"

# Step 2.4: Merge into Final JSON
echo ""
echo "[2.4] Merging into Final JSON..."
python "$MERGE_SCRIPT" \
    --input_dir "$CLEAN_OUTPUT_DIR" \
    --output_file "$FINAL_MERGED_FILE"

echo ""
echo "=================================================="
echo "Full Pipeline Complete!"
echo "Final Output: $FINAL_MERGED_FILE"
echo "=================================================="
