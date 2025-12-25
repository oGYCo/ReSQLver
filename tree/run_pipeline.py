import argparse
import json
import os
import sys
from typing import List
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .tree_builder import TreeBuilder
from .dpo_extractor import DPOExtractor
from .tree_node import RevisionTree

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def create_vllm_generator(model_path, tensor_parallel_size=1):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed. Please install it or use dummy mode.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        enforce_eager=True
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=1024,
        stop=["</answer>", "```sql\n\n"]
    )

    def generate(prompts: List[str]) -> List[str]:
        outputs = llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    return generate

def dummy_generator(prompts: List[str]) -> List[str]:
    """
    Dummy generator for testing logic without GPU.
    Returns a simple SQL query.
    """
    return ["```sql\nSELECT * FROM table LIMIT 1\n```"] * len(prompts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dev_bird_path", type=str, default="data/train_bird.json")
    parser.add_argument("--dev_json_path", type=str, default="data/train/train.json")
    parser.add_argument("--db_root_path", type=str, default="data/train/train_databases")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--sample_num", type=int, default=3)
    parser.add_argument("--use_dummy", action="store_true", help="Use dummy generator for testing")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of questions to process")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID for parallel processing")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--start_question_id", type=int, default=-1, help="Question ID to start from (resume)")
    parser.add_argument("--end_question_id", type=int, default=-1, help="Question ID to end at (inclusive)")
    
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    dev_bird = load_json(args.dev_bird_path)
    dev_json = load_json(args.dev_json_path)
    
    if len(dev_bird) != len(dev_json):
        print(f"Warning: dev_bird ({len(dev_bird)}) and dev_json ({len(dev_json)}) have different lengths.")
    
    # Sharding logic
    all_data = list(zip(dev_bird, dev_json))
    if args.num_shards > 1:
        chunk_size = (len(all_data) + args.num_shards - 1) // args.num_shards
        start_idx = args.shard_id * chunk_size
        end_idx = min((args.shard_id + 1) * chunk_size, len(all_data))
        all_data = all_data[start_idx:end_idx]
        print(f"Processing shard {args.shard_id}/{args.num_shards} (indices {start_idx}-{end_idx}) with {len(all_data)} examples.")

    # Initialize Generator
    if args.use_dummy:
        generator = dummy_generator
    else:
        generator = create_vllm_generator(args.model_path)

    # Initialize Tree Builder
    builder = TreeBuilder(generator, args.db_root_path, args.max_depth, args.sample_num)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_pairs = []
    
    # Process
    count = 0
    for i, (bird_item, json_item) in tqdm(enumerate(all_data), total=len(all_data)):
        if args.limit > 0 and count >= args.limit:
            break
            
        # Merge data
        data = json_item.copy()
        
        # Check resume and end
        if args.start_question_id >= 0 or args.end_question_id >= 0:
            current_qid = data.get("question_id")
            if current_qid is not None:
                try:
                    qid_int = int(current_qid)
                    if args.start_question_id >= 0 and qid_int < args.start_question_id:
                        continue
                    if args.end_question_id >= 0 and qid_int > args.end_question_id:
                        continue
                except (ValueError, TypeError):
                    pass

        data["input_seq"] = bird_item.get("input_seq", "")
        
        # Build Tree
        tree = builder.build_tree(data)
        
        # Save Tree
        tree_file = os.path.join(args.output_dir, f"tree_{data['question_id']}.json")
        with open(tree_file, 'w') as f:
            json.dump(tree.to_dict(), f, indent=2)
            
        # Extract Pairs
        pairs = DPOExtractor.extract_pairs(tree)
        all_pairs.extend(pairs)
        
        count += 1

    # Save all pairs
    pairs_file = os.path.join(args.output_dir, f"final_data_emerge_shard_{args.shard_id}.json")
    with open(pairs_file, 'w') as f:
        json.dump(all_pairs, f, indent=2)
        
    print(f"Done. Processed {count} questions. Generated {len(all_pairs)} pairs.")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
