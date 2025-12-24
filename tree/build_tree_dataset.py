"""
修订树数据集构建 - 主入口脚本

使用方法:
    python -m src.tree.build_tree_dataset \
        --model_path /path/to/model \
        --input_file /path/to/dev_bird.json \
        --db_path /path/to/dev_databases \
        --gold_file /path/to/dev.sql \
        --output_dir outputs/tree_dataset \
        --max_depth 5 \
        --sample_num 3
"""

import argparse
import os
import sys
import json
from typing import List, Dict, Any
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree.tree_node import RevisionTree
from tree.tree_builder import TreeBuilder, BatchTreeBuilder, ParallelBatchTreeBuilder
from tree.sql_executor import SQLExecutor, get_db_path
from tree.prompt_templates import PromptTemplates


def create_vllm_generator(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 8192,
    max_output_len: int = 2048,
    swap_space: int = 4
):
    """
    创建基于vLLM的模型生成器
    
    Args:
        model_path: 模型路径
        tensor_parallel_size: 张量并行大小
        gpu_memory_utilization: GPU内存使用率
        max_model_len: 最大模型长度
        max_output_len: 最大输出长度
        
    Returns:
        生成器函数
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoConfig
    
    print(f"Loading model from {model_path}...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 获取停止token
    stop_token_ids = config.eos_token_id if hasattr(config, "eos_token_id") else None
    if isinstance(stop_token_ids, int):
        stop_token_ids = [stop_token_ids]
    
    # 初始化LLM
    llm = LLM(
        model=model_path,
        dtype="float16",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=swap_space,
        enforce_eager=True,
        trust_remote_code=True
    )
    
    # 系统提示词
    system_prompt = PromptTemplates.SYSTEM_PROMPT
    
    def generate(prompts: List[str], temperature: float = 0.8) -> List[str]:
        """生成函数"""
        # 构建聊天消息
        chat_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            formatted = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            chat_prompts.append(formatted)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_output_len,
            n=1,
            stop_token_ids=stop_token_ids
        )
        
        # 生成
        outputs = llm.generate(chat_prompts, sampling_params=sampling_params)
        
        # 提取响应
        responses = []
        for output in outputs:
            response = output.outputs[0].text
            responses.append(response)
        
        return responses
    
    return generate


def create_mock_generator():
    """
    创建模拟生成器（用于测试）
    
    Returns:
        模拟生成器函数
    """
    import random
    
    def generate(prompts: List[str], temperature: float = 0.8) -> List[str]:
        """模拟生成函数"""
        responses = []
        for prompt in prompts:
            # 随机生成一个简单的SQL作为响应
            mock_sqls = [
                "SELECT * FROM table1",
                "SELECT id, name FROM users WHERE id = 1",
                "SELECT COUNT(*) FROM orders",
                "SELECT a.id, b.name FROM table_a a JOIN table_b b ON a.id = b.aid"
            ]
            sql = random.choice(mock_sqls)
            response = f"<think>\nLet me analyze the question...\n</think>\n<answer>\n```sql\n{sql}\n```\n</answer>"
            responses.append(response)
        return responses
    
    return generate


def load_dataset(input_file: str) -> List[Dict]:
    """加载数据集"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_dev_data(dev_file: str) -> List[Dict]:
    """加载dev.json数据"""
    with open(dev_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def merge_data(input_data: List[Dict], dev_data: List[Dict]) -> List[Dict]:
    """
    合并input_seq数据和dev数据
    
    Args:
        input_data: 包含input_seq的数据（dev_bird.json）
        dev_data: 原始dev数据（dev.json）
        
    Returns:
        合并后的数据
    """
    # 创建dev_data的索引
    dev_index = {}
    for item in dev_data:
        qid = item.get("question_id")
        if qid is not None:
            dev_index[qid] = item
    
    # 合并数据
    merged = []
    for idx, item in enumerate(input_data):
        # 获取question_id
        qid = item.get("question_id", idx)
        
        # 从dev_data获取额外信息
        dev_item = dev_index.get(qid, {})
        
        merged_item = {
            "question_id": qid,
            "db_id": item.get("db_id", dev_item.get("db_id", "")),
            "question": dev_item.get("question", ""),
            "evidence": dev_item.get("evidence", ""),
            "difficulty": dev_item.get("difficulty", "simple"),
            "SQL": dev_item.get("SQL", item.get("output_seq", "")),
            "input_seq": item.get("input_seq", "")
        }
        merged.append(merged_item)
    
    return merged


def main():
    parser = argparse.ArgumentParser(description="Build Revision Tree Dataset")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95,
                        help="GPU memory utilization")
    parser.add_argument("--swap_space", type=int, default=3,
                        help="CPU swap space size (GiB) per GPU")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for parallel tree building")
    
    # 数据参数
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input file (dev_bird.json)")
    parser.add_argument("--dev_file", type=str, default=None,
                        help="Path to dev.json file")
    parser.add_argument("--db_path", type=str, required=True,
                        help="Path to database directory")
    parser.add_argument("--gold_file", type=str, default=None,
                        help="Path to gold SQL file")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="outputs/tree_dataset",
                        help="Output directory")
    
    # 树构建参数
    parser.add_argument("--max_depth", type=int, default=5,
                        help="Maximum tree depth")
    parser.add_argument("--sample_num", type=int, default=3,
                        help="Number of samples per node")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="SQL execution timeout")
    parser.add_argument("--early_stop", action="store_true",
                        help="Stop when correct SQL found")
    
    # 其他参数
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="Ending index (-1 for all)")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save interval")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock generator for testing")
    parser.add_argument("--visible_devices", type=str, default="0,1,2,3",
                        help="Visible CUDA devices")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 60)
    print("Revision Tree Dataset Builder")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Input: {args.input_file}")
    print(f"Database: {args.db_path}")
    print(f"Output: {output_dir}")
    print(f"Max Depth: {args.max_depth}")
    print(f"Sample Num: {args.sample_num}")
    print("=" * 60)
    
    # 加载数据
    print("\nLoading data...")
    input_data = load_dataset(args.input_file)
    
    # 如果提供了dev_file，则合并数据
    if args.dev_file:
        dev_data = load_dev_data(args.dev_file)
        dataset = merge_data(input_data, dev_data)
    else:
        # 尝试自动推断dev.json路径
        dev_file = args.db_path.replace("_databases", ".json")
        if os.path.exists(dev_file):
            dev_data = load_dev_data(dev_file)
            dataset = merge_data(input_data, dev_data)
        else:
            dataset = input_data
    
    # 切片数据
    if args.end_idx > 0:
        dataset = dataset[args.start_idx:args.end_idx]
    elif args.start_idx > 0:
        dataset = dataset[args.start_idx:]
    
    print(f"Loaded {len(dataset)} questions")
    
    # 创建生成器
    print("\nInitializing model...")
    if args.mock:
        print("Using mock generator (for testing)")
        generator = create_mock_generator()
    else:
        generator = create_vllm_generator(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            swap_space=args.swap_space
        )
    
    # 创建批量构建器
    batch_builder = ParallelBatchTreeBuilder(
        model_generator=generator,
        db_root=args.db_path,
        output_dir=output_dir,
        max_depth=args.max_depth,
        sample_num=args.sample_num,
        temperature=args.temperature,
        timeout=args.timeout,
        early_stop=args.early_stop,
        save_interval=args.save_interval,
        resume=args.resume,
        batch_size=args.batch_size
    )
    
    # 提取input_seqs
    input_seqs = [item.get("input_seq", "") for item in dataset]
    
    # 构建树
    print("\nBuilding trees...")
    stats = batch_builder.build_trees(dataset, input_seqs)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Build Complete!")
    print("=" * 60)
    print(f"Total: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Good Trees: {stats['good_trees']}")
    print(f"Bad Trees: {stats['bad_trees']}")
    print(f"Skipped: {stats['skipped']}")
    
    if stats['processed'] > 0:
        success_rate = stats['good_trees'] / stats['processed'] * 100
        print(f"Success Rate: {success_rate:.2f}%")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
