# ReSQLver: SQL Reward Model Training & Evaluation Pipeline

ReSQLver 是一个用于训练和评估 SQL 生成任务奖励模型（Reward Model）的完整流水线。该项目基于 Qwen2.5-Coder 模型，通过构建 SQL 修正树（Revision Tree）来生成高质量的 DPO（Direct Preference Optimization）偏好数据，进而训练出能够准确评估 SQL 查询质量的奖励模型。

## 📋 目录

- [1. 环境搭建 (Environment Setup)](#1-环境搭建-environment-setup)
- [2. 数据准备 (Data Preparation)](#2-数据准备-data-preparation)
- [3. 模型准备 (Model Preparation)](#3-模型准备-model-preparation)
- [4. 构建树数据集与 DPO 数据 (Build Tree & DPO Dataset)](#4-构建树数据集与-dpo-数据-build-tree--dpo-dataset)
- [5. 奖励模型训练 (Reward Model Training)](#5-奖励模型训练-reward-model-training)
- [6. 模型评估 (Model Evaluation)](#6-模型评估-model-evaluation)
- [7. 验证树结构 (Optional Verification)](#7-验证树结构-optional-verification)

---

## 1. 环境搭建 (Environment Setup)

首先安装必要的系统工具和 Python 环境。建议使用 Conda 管理环境。

```bash
# 1. 安装系统依赖
sudo apt update
sudo apt install -y tmux git wget

# 2. 创建并激活 Conda 环境
conda create -n resqlver python=3.12 -y
conda activate resqlver

# 3. 安装 Python 依赖
pip install -r requirements.txt
```

**核心依赖：**
- `vllm`: 用于高效的大模型推理。
- `transformers`, `peft`, `trl`: 用于模型训练和微调。
- `sqlite3`: 用于执行和验证 SQL 查询。

---

## 2. 数据准备 (Data Preparation)

本项目使用 BIRD (BIg Bench for Large-scale Database Grounded Text-to-SQL Evaluation) 数据集作为训练基础。

```bash
# 1. 进入 data 目录下载数据
cd data
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
unzip train.zip
cd train
unzip train_database
cd ..
cd ..

# 2. 数据预处理（添加 question_id）
# 这一步会为 BIRD 数据集添加唯一的 question_id，方便后续追踪
python add_question_id.py
```

---

## 3. 模型准备 (Model Preparation)

下载 Qwen2.5-Coder 模型作为基座模型。可以使用提供的脚本或手动下载。

```bash
bash download_model.sh
```

或者使用 `huggingface-cli` 手动下载：

```bash
# 下载 1.5B 模型（用于训练奖励模型）
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir Qwen2.5-Coder-1.5B-Instruct

# 下载 3B 模型（用于生成树数据）
huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct --local-dir Qwen2.5-Coder-3B-Instruct
```

---

## 4. 构建树数据集与 DPO 数据 (Build Tree & DPO Dataset)

这是本项目的核心部分。我们通过让模型自我修正错误的 SQL 查询，构建一个“修正树”（Revision Tree）。树中的节点代表 SQL 查询，边代表修正过程。通过比较树中不同节点的质量（是否正确、修正距离等），我们可以提取出高质量的 DPO 偏好对（Chosen vs Rejected）。

### 4.1 运行并行生成脚本

使用 `tree/run_parallel.sh` 脚本并行生成数据。该脚本会自动利用多张 GPU 进行加速。

**配置参数 (`tree/run_parallel.sh`):**
- `NUM_GPUS`: 并行使用的 GPU 数量。
- `MODEL_PATH`: 用于生成的模型路径 (推荐 `Qwen2.5-Coder-3B-Instruct`)。
- `OUTPUT_DIR`: 输出目录 (默认为 `output`)。

**运行命令：**

```bash
# 确保在项目根目录下
bash tree/run_parallel.sh
```

### 4.2 合并数据分片

并行运行结束后，`output` 目录下会生成多个 `final_data_emerge_shard_*.json` 文件。使用以下 Python 脚本将它们合并为一个完整的训练文件：

```python
import json
import glob

# 查找所有分片文件
output_files = glob.glob("output/final_data_emerge_shard_*.json")
all_data = []

# 读取并合并
for f in output_files:
    with open(f, 'r') as fp:
        all_data.extend(json.load(fp))

# 保存合并后的文件
with open("final_data_train.json", "w") as f:
    json.dump(all_data, f, indent=2)

print(f"Merged {len(all_data)} pairs to final_data_train.json")
```

---

## 5. 奖励模型训练 (Reward Model Training)

使用生成的 DPO 数据训练奖励模型。我们使用 LoRA (Low-Rank Adaptation) 技术进行高效微调。

### 5.1 配置训练脚本

打开 `train.py`，根据实际情况修改以下配置：

```python
# train.py
MODEL_ID = "Qwen2.5-Coder-1.5B-Instruct"  # 基座模型路径
DATA_PATH = "final_data_train.json"       # 上一步合并得到的数据集路径
OUTPUT_DIR = "qwen-reward-model-sql-train" # 输出模型路径
```

### 5.2 开始训练

```bash
python train.py
```

训练过程中，模型会学习区分高质量（Chosen）和低质量（Rejected）的 SQL 查询。训练完成后，模型权重将保存在 `qwen-reward-model-sql-train` 目录中。

---

## 6. 模型评估 (Model Evaluation)

评估训练好的奖励模型在测试集上的表现。

```bash
python evaluate.py \
    --base_model Qwen2.5-Coder-1.5B-Instruct \
    --adapters qwen-reward-model-sql-train \
    --test_data final_data_test.json \
    --batch_size 64
```

**参数说明:**
- `--base_model`: 基座模型路径。
- `--adapters`: 训练好的 LoRA 适配器路径。
- `--test_data`: 测试数据集路径。
- `--batch_size`: 批处理大小。

---

## 7. 验证树结构 (Optional Verification)

如果需要验证生成的树结构是否合法，或者检查生成的 SQL 是否能正确执行，可以使用 `eval` 模块生成详细的验证报告。

```bash
python -m eval \
    --tree_file output/tree_dataset/partX/timestamp/all_trees.json \
    --db_root data/train/train_databases \
    --output verification_report.json
```

---

## 📁 项目结构

```
ReSQLver/
├── data/                   # 数据集目录
├── output/                 # 生成结果输出目录
├── tree/                   # 树构建与数据提取核心代码
│   ├── tree_builder.py     # 树构建逻辑
│   ├── tree_node.py        # 树节点定义
│   ├── dpo_extractor.py    # DPO 数据提取逻辑
│   ├── sql_utils.py        # SQL 执行与验证工具
│   ├── prompts.py          # Prompt 模板
│   ├── run_pipeline.py     # 单个 Pipeline 运行入口
│   └── run_parallel.sh     # 并行运行脚本
├── train.py                # 奖励模型训练脚本
├── evaluate.py             # 模型评估脚本
├── add_question_id.py      # 数据预处理脚本
├── download_model.sh       # 模型下载脚本
├── requirements.txt        # 依赖列表
└── README.md               # 项目文档
```
