import torch
from datasets import load_dataset, Dataset
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# 1. 配置路径和参数
MODEL_ID = "Qwen2.5-Coder-1.5B-Instruct"
DATA_PATH = "baseline.json"
OUTPUT_DIR = "qwen-reward-model-sql-base"
LEARNING_RATE = 5e-6 

def main():
    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. 准备数据并切分 (修改点)
    print("正在加载数据...")
    full_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    # --- [关键修改] 随机切分 5% 作为验证集 ---
    # seed=42 保证每次切分结果一致
    dataset_split = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    print(f"数据加载完成。")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    # ------------------------------------------------

    # 4. 数据格式化函数
    def formatting_func(examples):
        new_examples = {"chosen": [], "rejected": []}
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            new_examples["chosen"].append([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen}
            ])
            new_examples["rejected"].append([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected}
            ])
        return new_examples

    # 批量应用格式转换
    # 注意：需要分别处理 train 和 eval
    column_names = train_dataset.column_names
    
    print("正在格式化训练集...")
    train_formatted = train_dataset.map(
        formatting_func, 
        batched=True, 
        remove_columns=column_names,
        num_proc=8 # 多进程加速处理
    )
    
    print("正在格式化验证集...")
    eval_formatted = eval_dataset.map(
        formatting_func, 
        batched=True, 
        remove_columns=column_names,
        num_proc=4
    )

    # 5. 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"] 
    )
    
    # 6. 加载模型
    torch.backends.cuda.enable_flash_sdp(True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=1, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).cuda()
    model.config.pad_token_id = tokenizer.pad_token_id

    # 7. 配置 RewardConfig (A100 80G 激进优化版)
    training_args = RewardConfig(
        output_dir=OUTPUT_DIR,
        
        # --- [关键修改] 显存优化参数 ---
        # 1.5B 模型很小，A100 80G 可以容纳巨大的 Batch Size
        # 如果 64 仍然显存不满，可以尝试 96 或 128
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2, # batch很大了，不需要累积
        
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        max_length=4096, 
        num_train_epochs=1, 
        
        # --- 验证策略配置 ---
        eval_strategy="steps",   # 按步数评估
        eval_steps=50,           # 每50步评估一次
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
        
        # --- 性能加速 ---
        bf16=True,
        tf32=True,               # [重要] A100 开启 TF32 加速矩阵运算
        dataloader_num_workers=8,# [重要] 增加数据加载线程，防止 GPU 等待 CPU
        dataloader_pin_memory=True,
        
        report_to="none", 
        center_rewards_coefficient=0.01,
        remove_unused_columns=False,
        gradient_checkpointing=True, # 1.5B模型在80G卡上通常不需要这个，关掉可以跑得更快
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 8. 初始化 Trainer
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_formatted,
        eval_dataset=eval_formatted, # 传入验证集
        peft_config=peft_config,
    )

    # 9. 开始训练
    print("开始训练奖励模型...")
    train_result = trainer.train()

    # 10. 保存模型
    print(f"训练完成，保存模型至 {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 11. 保存指标 (包含验证集结果)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # 绘制图表逻辑 (保持原样，增加 eval 曲线的兼容性)
    try:
        import matplotlib.pyplot as plt
        import json
        
        state_path = os.path.join(OUTPUT_DIR, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            log_history = state.get("log_history", [])
            
            # 分离 train 和 eval 数据
            train_steps, train_loss = [], []
            eval_steps, eval_loss = [], []
            eval_acc = []
            
            for entry in log_history:
                if "loss" in entry and "step" in entry:
                    train_steps.append(entry["step"])
                    train_loss.append(entry["loss"])
                if "eval_loss" in entry:
                    eval_steps.append(entry["step"])
                    eval_loss.append(entry["eval_loss"])
                    # 尝试获取验证集准确率
                    if "eval_rewards/accuracies" in entry:
                        eval_acc.append(entry["eval_rewards/accuracies"])

            # 绘制 Loss
            plt.figure(figsize=(10, 5))
            plt.plot(train_steps, train_loss, label='Training Loss')
            if eval_loss:
                plt.plot(eval_steps, eval_loss, label='Validation Loss', linestyle='--')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curve')
            plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
            plt.close()

            print(f"图表已更新，包含验证集曲线。")

    except Exception as e:
        print(f"绘图错误: {e}")

if __name__ == "__main__":
    main()