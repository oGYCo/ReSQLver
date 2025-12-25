# ReSQLver: SQL Reward Model Training & Evaluation Pipeline

本文档详细介绍了从环境搭建、数据准备、树数据集构建、DPO数据提取、奖励模型训练到最终评估的完整流程。

## 1. 环境搭建 (Environment Setup)

首先安装必要的系统工具和 Python 环境。

\`\`\`bash
# 1. 安装系统依赖
sudo apt update
sudo apt install -y tmux git wget

# 2. 创建并激活 Conda 环境
conda create -n vllm python=3.12 -y
conda activate vllm

# 3. 安装 Python 依赖
pip install -r requirements.txt
\`\`\`

## 2. 数据准备 (Data Preparation)

下载 BIRD 数据集并进行预处理。

\`\`\`bash
# 1. 进入 data 目录下载数据
cd data
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
unzip train.zip
cd train
unzip train_database
cd ..
cd ..

# 2. 数据预处理（添加 question_id）
python add_question_id.py
\`\`\`

## 3. 模型准备 (Model Preparation)

下载 Qwen2.5-Coder 模型。可以使用提供的脚本或手动下载。

\`\`\`bash
bash download_model.sh
# 或者使用 huggingface-cli 下载
# huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir Qwen2.5-Coder-1.5B-Instruct
# huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct --local-dir Qwen2.5-Coder-3B-Instruct
\`\`\`

## 4. 构建树数据集与 DPO 数据 (Build Tree & DPO Dataset)

使用并行脚本高效构建树数据集。该过程会自动生成用于 DPO 训练的偏好对数据。

### 4.1 运行并行生成脚本

修改 `tree/run_parallel.sh` 中的配置（如 GPU 数量、模型路径等），然后运行：

\`\`\`bash
# 确保在项目根目录下
bash tree/run_parallel.sh
\`\`\`

**参数说明 (`tree/run_parallel.sh`):**
- `NUM_GPUS`: 使用的 GPU 数量。
- `MODEL_PATH`: 模型路径 (例如 `Qwen2.5-Coder-3B-Instruct`)。
- `OUTPUT_DIR`: 输出目录。

### 4.2 合并数据分片

并行运行结束后，`output` 目录下会生成多个 `final_data_emerge_shard_*.json` 文件。需要将它们合并为一个训练文件（例如 `final_data_train.json`）。

\`\`\`python
import json
import glob

output_files = glob.glob("output/final_data_emerge_shard_*.json")
all_data = []
for f in output_files:
    with open(f, 'r') as fp:
        all_data.extend(json.load(fp))

with open("final_data_train.json", "w") as f:
    json.dump(all_data, f, indent=2)
print(f"Merged {len(all_data)} pairs to final_data_train.json")
\`\`\`

## 5. 奖励模型训练 (Reward Model Training)

使用生成的 DPO 数据训练奖励模型。

### 5.1 配置训练脚本

打开 `train.py`，根据实际情况修改以下配置：

\`\`\`python
# train.py
MODEL_ID = "Qwen2.5-Coder-1.5B-Instruct"  # 基座模型路径
DATA_PATH = "final_data_train.json"       # 上一步合并得到的数据集路径
OUTPUT_DIR = "qwen-reward-model-sql-train" # 输出模型路径
\`\`\`

### 5.2 开始训练

\`\`\`bash
python train.py
\`\`\`

训练完成后，模型权重将保存在 `qwen-reward-model-sql-train` 目录中。

## 6. 模型评估 (Model Evaluation)

评估训练好的奖励模型在测试集上的表现。

\`\`\`bash
python evaluate.py \
    --base_model Qwen2.5-Coder-1.5B-Instruct \
    --adapters qwen-reward-model-sql-train \
    --test_data final_data_test.json \
    --batch_size 64
\`\`\`

**参数说明:**
- `--base_model`: 基座模型路径。
- `--adapters`: 训练好的 LoRA 适配器路径。
- `--test_data`: 测试数据集路径。
- `--batch_size`: 批处理大小。

## 7. 验证树结构 (Optional Verification)

如果需要验证生成的树结构是否合法，可以使用 `eval` 模块生成报告。

\`\`\`bash
python -m eval \
    --tree_file output/tree_dataset/partX/timestamp/all_trees.json \
    --db_root data/train/train_databases \
    --output verification_report.json
\`\`\`
