import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import load_dataset
import argparse
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# 定义一个处理函数，用于批量 Tokenize
def preprocess_function(examples, tokenizer, max_length=4096):
    # 构建 Chat 模版
    chosen_texts = []
    rejected_texts = []
    
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        chosen_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}
        ]
        rejected_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected}
        ]
        # 应用模版但不进行 tokenize (转为纯文本)
        chosen_texts.append(tokenizer.apply_chat_template(chosen_messages, tokenize=False))
        rejected_texts.append(tokenizer.apply_chat_template(rejected_messages, tokenize=False))
    
    # 批量 Tokenize
    # padding=True 会自动 padding 到当前 batch 中最长的序列，比 padding='max_length' 更省显存
    tokenized_chosen = tokenizer(chosen_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    tokenized_rejected = tokenizer(rejected_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }

# 自定义 Collate 函数，用于 DataLoader 组装 batch
def collate_fn(batch):
    # batch 是一个 list of dict，我们需要将其转换为 dict of tensor
    # 这里我们利用 tokenizer 在 preprocess 阶段已经做好的 tensor，只需要 stack 或者 pad
    # 但为了简单和通用，我们在 DataLoader 循环里做 padding 更灵活，
    # 或者为了最高效，我们在 preprocess 里直接不做 padding，在这里做 padding。
    
    # 简化策略：为了代码清晰，我们采用在循环中动态 padding 的策略，
    # 所以这里的 collate_fn 只是简单的透传 list，实际处理在 evaluation 循环内部的 tokenizer 再次调用
    # 或者使用更高效的方法：见下文 evaluate_one_model 中的实现
    return batch

def evaluate_one_model(base_model_path, adapter_path, dataset, batch_size=32, device="cuda"):
    print(f"\n{'='*50}")
    print(f"Evaluating Adapter: {adapter_path}")
    print(f"{'='*50}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 重要：设置 padding 方向，对于分类模型，通常右侧 padding 是安全的
    tokenizer.padding_side = "right" 

    # 2. Load Base Model
    # [关键修改] 添加 attn_implementation="sdpa" 解决报错并加速
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # <--- 修复报错的关键
        device_map=device
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 3. Load LoRA Adapter
    if adapter_path and adapter_path.lower() != "none":
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✅ LoRA adapter loaded successfully.")
    else:
        print("⚠️ No adapter loaded. Evaluating base model (random score head)!")

    model.eval()

    correct_count = 0
    total_count = 0
    results = []

    # 4. 使用 DataLoader 进行批量推理
    # 由于文本长度不一，我们不使用 dataset.map 预处理所有 tensor (太占内存)，
    # 而是直接在循环中批量 tokenize
    print(f"Processing {len(dataset)} samples with batch_size={batch_size}...")
    
    # 创建 DataLoader，num_workers 可以加速数据读取
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            prompts = batch["prompt"]
            chosens = batch["chosen"]
            rejecteds = batch["rejected"]
            
            # --- 批量构建文本 ---
            chosen_texts = []
            rejected_texts = []
            for p, c, r in zip(prompts, chosens, rejecteds):
                chosen_texts.append(tokenizer.apply_chat_template([
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": c}
                ], tokenize=False))
                rejected_texts.append(tokenizer.apply_chat_template([
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": r}
                ], tokenize=False))
            
            # --- 批量 Tokenize (Padding 到当前 Batch 最长) ---
            inputs_chosen = tokenizer(
                chosen_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
            ).to(device)
            
            inputs_rejected = tokenizer(
                rejected_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
            ).to(device)
            
            # --- 批量推理 ---
            # 这里的输出 shape 是 [batch_size, 1]
            rewards_chosen = model(**inputs_chosen).logits.squeeze(-1)
            rewards_rejected = model(**inputs_rejected).logits.squeeze(-1)
            
            # --- 统计结果 ---
            # 比较 rewards_chosen > rewards_rejected
            # current_correct 是一个布尔 Tensor [True, False, ...]
            current_correct = (rewards_chosen > rewards_rejected)
            correct_count += current_correct.sum().item()
            total_count += len(prompts)
            
            # 收集详细结果 (为了避免内存溢出，我们只存简单的 float)
            margins = (rewards_chosen - rewards_rejected).tolist()
            chosen_scores = rewards_chosen.tolist()
            rejected_scores = rewards_rejected.tolist()
            is_correct_list = current_correct.tolist()
            
            for i in range(len(prompts)):
                results.append({
                    "score_chosen": chosen_scores[i],
                    "score_rejected": rejected_scores[i],
                    "margin": margins[i],
                    "is_correct": is_correct_list[i]
                })

    accuracy = correct_count / total_count
    
    # Calculate average margin
    avg_margin = sum(r["margin"] for r in results) / len(results)
    
    print(f"\nResult for {os.path.basename(adapter_path)}:")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    print(f"Avg Margin: {avg_margin:.4f}")
    
    # 清理显存
    del model
    torch.cuda.empty_cache()

    return {
        "adapter": adapter_path,
        "accuracy": accuracy,
        "avg_margin": avg_margin,
        # "details": results # 如果数据量太大，建议注释掉这一行，否则 json 文件会巨大
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Reward Models")
    parser.add_argument("--base_model", type=str, default="Qwen2.5-Coder-1.5B-Instruct", help="Path to base model")
    # 可以在命令行传入多个 adapter 路径，或者写 None 来测试基座
    parser.add_argument("--adapters", type=str, nargs='+', default=["qwen-reward-model-sql-train"], help="List of adapter paths")
    parser.add_argument("--test_data", type=str, default="final_data_test.json", help="Path to test dataset")
    parser.add_argument("--output_file", type=str, default="eval_results.json", help="Path to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    # Load Dataset
    print(f"Loading test data from {args.test_data}...")
    # 注意：这里我们随机切分出 5% 作为测试集，保持和训练时一致的逻辑，或者你可以指定单独的测试文件
    full_dataset = load_dataset("json", data_files=args.test_data, split="test")
    
    # 如果你想评估之前的验证集，可以使用相同的 seed 切分
    dataset = full_dataset

    all_metrics = []

    for adapter_path in args.adapters:
        metrics = evaluate_one_model(args.base_model, adapter_path, dataset, batch_size=args.batch_size)
        all_metrics.append(metrics)

    # Print Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<40} | {'Accuracy':<10} | {'Avg Margin':<10}")
    print("-" * 64)
    for m in all_metrics:
        name = os.path.basename(m["adapter"])
        print(f"{name:<40} | {m['accuracy']:.2%}    | {m['avg_margin']:.4f}")
    print("="*60)

    with open(args.output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Detailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()