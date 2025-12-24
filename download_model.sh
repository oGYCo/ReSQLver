#!/bin/bash

# 获取脚本所在目录的上一级目录（即工作区根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Workspace Root: $WORKSPACE_ROOT"

# 1. 安装 huggingface_hub
pip install -U huggingface_hub

# 2. 下载 Qwen2.5-Coder-3B-Instruct 模型
# 使用 hf-mirror 加速下载（可选）
export HF_ENDPOINT=https://hf-mirror.com

echo "Downloading Qwen2.5-Coder-3B-Instruct..."
huggingface-cli download --resume-download Qwen/Qwen2.5-Coder-3B-Instruct \
    --local-dir "$WORKSPACE_ROOT/Qwen2.5-Coder-3B-Instruct" \
    --local-dir-use-symlinks False

echo "Model downloaded to: $WORKSPACE_ROOT/Qwen2.5-Coder-3B-Instruct"