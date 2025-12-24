import json
import os
from tqdm import tqdm


def load_comparison_tree(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容被 list 包一层的情况
    if isinstance(data, list):
        data = data[0]

    return data


def extract_dpo_from_comparison_tree(tree):
    """
    从 comparison tree 中提取 DPO 格式数据
    """
    prompt = tree.get("question", "")
    comparisons = tree.get("comparisons", {})

    results = []

    for _, comp in comparisons.items():
        chosen_id = comp.get("chosen")
        if not chosen_id:
            continue

        # 除了 "chosen"，其余 key 都是 node_id
        node_ids = [k for k in comp.keys() if k != "chosen"]
        if len(node_ids) != 2:
            continue

        node_a, node_b = node_ids

        if chosen_id == node_a:
            chosen_node = comp[node_a]
            rejected_node = comp[node_b]
        elif chosen_id == node_b:
            chosen_node = comp[node_b]
            rejected_node = comp[node_a]
        else:
            continue

        chosen_sql = chosen_node.get("generated_sql", "")
        rejected_sql = rejected_node.get("generated_sql", "")

        # 防御：空 SQL 不要
        if not chosen_sql or not rejected_sql:
            continue

        results.append({
            "prompt": prompt,
            "chosen": chosen_sql,
            "rejected": rejected_sql
        })

    return results


import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract DPO data from comparison trees")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing comparison JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for DPO JSON files")
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    total_pairs = 0
    processed = 0

    for fname in tqdm(files, desc="Processing comparison trees"):
        in_path = os.path.join(INPUT_DIR, fname)

        try:
            tree = load_comparison_tree(in_path)
            dpo_data = extract_dpo_from_comparison_tree(tree)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")
            continue

        if not dpo_data:
            continue

        qid = tree.get("question_id", os.path.splitext(fname)[0])
        out_name = f"dpo_{qid}.json"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dpo_data, f, indent=2, ensure_ascii=False)

        processed += 1
        total_pairs += len(dpo_data)

    print("\n========== SUMMARY ==========")
    print(f"Total files processed: {processed}")
    print(f"Total DPO pairs: {total_pairs}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
