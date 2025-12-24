#!/bin/bash
# ==========================================================
# 修订树数据集构建脚本 (Train Dataset)
# ==========================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$(dirname "$PROJECT_ROOT")/data"

# 配置参数
MODEL_PATH="/home/taocy/yuchen/Qwen2.5-Coder-1.5B-Instruct"  # 模型路径
INPUT_FILE="${DATA_ROOT}/train_bird.json"           # 输入文件 (train_bird.json)
DEV_FILE="${DATA_ROOT}/train/train.json"            # dev文件 (这里用train.json)
DB_PATH="${DATA_ROOT}/train/train_databases"        # 数据库目录 (train_databases)
OUTPUT_DIR="${PROJECT_ROOT}/outputs/tree_dataset_train" # 输出目录

# 树构建参数
MAX_DEPTH=5          # 最大深度
SAMPLE_NUM=3         # 每个节点的采样数
TEMPERATURE=0.8      # 采样温度
TIMEOUT=30           # SQL执行超时时间（秒）

# GPU设置
VISIBLE_DEVICES="0"  # 可见的GPU设备
TENSOR_PARALLEL=1    # 张量并行大小
GPU_MEM_UTIL=0.95    # GPU内存使用率

# 数据范围（可选）
START_IDX=0          # 起始索引
END_IDX=-1           # 结束索引 (-1表示全部)

# 其他选项
SAVE_INTERVAL=10     # 保存间隔
EARLY_STOP="--early_stop" # 是否在找到正确SQL后停止 (设为"--early_stop"启用)
RESUME=""            # 是否从检查点恢复 (设为"--resume"启用)
MOCK=""              # 是否使用mock模式测试 (设为"--mock"启用)

# ==========================================================
# 主程序
# ==========================================================

cd "$PROJECT_ROOT" || exit

echo "=================================================="
echo "Revision Tree Dataset Builder (Train Set)"
echo "=================================================="
echo "Model: ${MODEL_PATH}"
echo "Input: ${INPUT_FILE}"
echo "Dev/Train File: ${DEV_FILE}"
echo "Database: ${DB_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Max Depth: ${MAX_DEPTH}"
echo "Sample Num: ${SAMPLE_NUM}"
echo "Temperature: ${TEMPERATURE}"
echo "=================================================="

# 运行Python脚本
python -m tree.build_tree_dataset \
    --model_path "${MODEL_PATH}" \
    --input_file "${INPUT_FILE}" \
    --dev_file "${DEV_FILE}" \
    --db_path "${DB_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_depth ${MAX_DEPTH} \
    --sample_num ${SAMPLE_NUM} \
    --temperature ${TEMPERATURE} \
    --timeout ${TIMEOUT} \
    --tensor_parallel_size ${TENSOR_PARALLEL} \
    --gpu_memory_utilization ${GPU_MEM_UTIL} \
    --visible_devices "${VISIBLE_DEVICES}" \
    --start_idx ${START_IDX} \
    --end_idx ${END_IDX} \
    --save_interval ${SAVE_INTERVAL} \
    ${EARLY_STOP} \
    ${RESUME} \
    ${MOCK}

echo ""
echo "Build complete!"
