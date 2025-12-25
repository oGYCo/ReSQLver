#!/bin/bash

# Set paths
MODEL_PATH="Qwen2.5-Coder-3B-Instruct"
DEV_BIRD="data/train_bird.json"
DEV_JSON="data/train/train.json"
DB_ROOT="data/train/train_databases"
OUTPUT_DIR="output"

# Run pipeline
python3 -m tree.run_pipeline \
    --model_path "$MODEL_PATH" \
    --dev_bird_path "$DEV_BIRD" \
    --dev_json_path "$DEV_JSON" \
    --db_root_path "$DB_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --max_depth 5 \
    --sample_num 2
