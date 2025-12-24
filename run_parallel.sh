#!/bin/bash

# Get the directory where the script is located (ReSQLver directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# The workspace root (parent of ReSQLver)
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
MODEL_PATH="$WORKSPACE_ROOT/Qwen2.5-Coder-3B-Instruct"
INPUT_FILE="$WORKSPACE_ROOT/data/train_bird.json"
DB_PATH="$WORKSPACE_ROOT/data/train/train_databases"
GOLD_FILE="$WORKSPACE_ROOT/data/train/train_gold.sql"
OUTPUT_BASE="$WORKSPACE_ROOT/ReSQLver/outputs/train_tree_dataset"

# Python script path (Absolute path)
PYTHON_SCRIPT="$SCRIPT_DIR/tree/build_tree_dataset.py"

# Create logs directory in workspace root
mkdir -p "$WORKSPACE_ROOT/logs"

echo "Starting parallel generation on 4 GPUs..."
echo "Workspace Root: $WORKSPACE_ROOT"
echo "Python Script: $PYTHON_SCRIPT"

# GPU 0 (Items 0-384)
echo "Launching GPU 0 job..."
nohup python "$PYTHON_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --db_path "$DB_PATH" \
    --gold_file "$GOLD_FILE" \
    --output_dir "${OUTPUT_BASE}/part0" \
    --start_idx 338 --end_idx 384 \
    --max_depth 5 \
    --sample_num 2 \
    --tensor_parallel_size 1 \
    --visible_devices 0 \
    > "$WORKSPACE_ROOT/ReSQLver/outputs/gpu0.log" 2>&1 &

# GPU 1 (Items 384-768)
# echo "Launching GPU 1 job..."
# nohup python "$PYTHON_SCRIPT" \
#     --model_path "$MODEL_PATH" \
#     --input_file "$INPUT_FILE" \
#     --db_path "$DB_PATH" \
#     --gold_file "$GOLD_FILE" \
#     --output_dir "${OUTPUT_BASE}/part1" \
#     --start_idx 421 --end_idx 768 \
#     --max_depth 5 \
#     --sample_num 2 \
#     --tensor_parallel_size 1 \
#     --visible_devices 1 \
#     > "$WORKSPACE_ROOT/ReSQLver/outputs/gpu1.log" 2>&1 &

# GPU 2 (Items 768-1152)
# echo "Launching GPU 2 job..."
# nohup python "$PYTHON_SCRIPT" \
#     --model_path "$MODEL_PATH" \
#     --input_file "$INPUT_FILE" \
#     --db_path "$DB_PATH" \
#     --gold_file "$GOLD_FILE" \
#     --output_dir "${OUTPUT_BASE}/part2" \
#     --start_idx 768 --end_idx 1152 \
#     --max_depth 5 \
#     --sample_num 2 \
#     --tensor_parallel_size 1 \
#     --visible_devices 2 \
#     > "$WORKSPACE_ROOT/ReSQLver/outputs/gpu2.log" 2>&1 &

# GPU 3 (Items 1152-1534)
# echo "Launching GPU 3 job..."
# nohup python "$PYTHON_SCRIPT" \
#     --model_path "$MODEL_PATH" \
#     --input_file "$INPUT_FILE" \
#     --db_path "$DB_PATH" \
#     --gold_file "$GOLD_FILE" \
#     --output_dir "${OUTPUT_BASE}/part3" \
#     --start_idx 1224 --end_idx 1534 \
#     --max_depth 5 \
#     --sample_num 2 \
#     --tensor_parallel_size 1 \
#     --visible_devices 3 \
#     > "$WORKSPACE_ROOT/ReSQLver/outputs/gpu3.log" 2>&1 &

echo "All jobs launched. Check logs/gpu*.log for progress."
