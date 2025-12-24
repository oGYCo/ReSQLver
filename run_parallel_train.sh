#!/bin/bash

# Get the directory where the script is located (ReSQLver directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# The workspace root (parent of ReSQLver)
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
MODEL_PATH="$WORKSPACE_ROOT/Qwen2.5-Coder-3B-Instruct"
INPUT_FILE="$WORKSPACE_ROOT/data/train_bird.json"
DEV_FILE="$WORKSPACE_ROOT/data/train/train.json"
DB_PATH="$WORKSPACE_ROOT/data/train/train_databases"
GOLD_FILE="$WORKSPACE_ROOT/data/train/train_gold.sql"
OUTPUT_BASE="$WORKSPACE_ROOT/ReSQLver/outputs/train_tree_dataset_full"

# Python script path (Absolute path)
PYTHON_SCRIPT="$SCRIPT_DIR/tree/build_tree_dataset.py"

# Create logs directory in workspace root
mkdir -p "$WORKSPACE_ROOT/logs"
mkdir -p "$OUTPUT_BASE"

echo "Starting parallel generation on 4 GPUs for TRAIN dataset (9428 items)..."
echo "Workspace Root: $WORKSPACE_ROOT"
echo "Python Script: $PYTHON_SCRIPT"

# Total items: 9428
# Split into 4 parts: ~2357 items per GPU

# GPU 0 (Items 0-2357)
echo "Launching GPU 0 job (0-2357)..."
nohup python "$PYTHON_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --dev_file "$DEV_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${OUTPUT_BASE}/part0" \
    --start_idx 0 --end_idx 2357 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 0 \
    --early_stop \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/train_gpu0.log" 2>&1 &

# GPU 1 (Items 2357-4714)
echo "Launching GPU 1 job (2357-4714)..."
nohup python "$PYTHON_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --dev_file "$DEV_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${OUTPUT_BASE}/part1" \
    --start_idx 2357 --end_idx 4714 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 1 \
    --early_stop \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/train_gpu1.log" 2>&1 &

# GPU 2 (Items 4714-7071)
echo "Launching GPU 2 job (4714-7071)..."
nohup python "$PYTHON_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --dev_file "$DEV_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${OUTPUT_BASE}/part2" \
    --start_idx 4714 --end_idx 7071 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 2 \
    --early_stop \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/train_gpu2.log" 2>&1 &

# GPU 3 (Items 7071-9428)
echo "Launching GPU 3 job (7071-9428)..."
nohup python "$PYTHON_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --dev_file "$DEV_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${OUTPUT_BASE}/part3" \
    --start_idx 7071 --end_idx 9428 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 3 \
    --early_stop \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/train_gpu3.log" 2>&1 &

echo "All jobs launched. Check ReSQLver/outputs/train_gpu*.log for progress."
