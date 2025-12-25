import os
import subprocess
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--script", type=str, default="tree.run_pipeline", help="Script module to run")
    parser.add_argument("--model_path", type=str, default="Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dev_bird_path", type=str, default="data/train_bird.json")
    parser.add_argument("--dev_json_path", type=str, default="data/train/train.json")
    parser.add_argument("--db_root_path", type=str, default="data/train/train_databases")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--sample_num", type=int, default=2)
    parser.add_argument("--start_question_id", type=int, default=-1, help="Question ID to start from (resume)")
    parser.add_argument("--end_question_id", type=int, default=-1, help="Question ID to end at (inclusive)")
    
    args = parser.parse_args()

    processes = []
    
    for i in range(args.num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        
        cmd = [
            "python3", "-m", args.script,
            "--model_path", args.model_path,
            "--dev_bird_path", args.dev_bird_path,
            "--dev_json_path", args.dev_json_path,
            "--db_root_path", args.db_root_path,
            "--output_dir", args.output_dir,
            "--max_depth", str(args.max_depth),
            "--sample_num", str(args.sample_num),
            "--shard_id", str(i),
            "--num_shards", str(args.num_gpus),
            "--start_question_id", str(args.start_question_id),
            "--end_question_id", str(args.end_question_id)
        ]
        
        print(f"Starting process for GPU {i} (Shard {i}/{args.num_gpus})...")
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        
        # Stagger start times slightly to avoid race conditions on file creation or other resources
        time.sleep(5)

    print(f"Launched {len(processes)} processes. Waiting for completion...")
    
    for p in processes:
        p.wait()
        
    print("All processes completed.")

if __name__ == "__main__":
    main()
