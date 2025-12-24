#!/bin/bash
sudo apt update
sudo apt install -y tmux git wget
cd data
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
unzip train.zip
cd ..
bash download_model.sh
conda create -n vllm python=3.12 -y
conda activate vllm
pip install -r requirements.txt
python add_question_id.py
bash run_full_pipeline.sh
