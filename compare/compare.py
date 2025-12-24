import json
import os
from datetime import datetime
from tqdm import tqdm


# -------------------- I/O helpers --------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# -------------------- comparison logic --------------------
def maybe_add_comparison(a, b, comparisons):
    a_correct = a.get("is_correct", False)
    b_correct = b.get("is_correct", False)

    # Rule 1: 一真一假
    if a_correct != b_correct:
        chosen = a if a_correct else b
        key = f"compare success:{a['node_id']},{b['node_id']} ，choose {chosen['node_id']}"
        comparisons[key] = {
            a["node_id"]: a,
            b["node_id"]: b,
            "chosen": chosen["node_id"]
        }
        return key

    # Rule 2: both wrong, revision_distance !=
    if not a_correct and not b_correct:
        da = a.get("revision_distance", 999)
        db = b.get("revision_distance", 999)
        if da != db:
            chosen = a if da < db else b
            key = f"compare success:{a['node_id']},{b['node_id']} ，choose {chosen['node_id']}"
            comparisons[key] = {
                a["node_id"]: a,
                b["node_id"]: b,
                "chosen": chosen["node_id"]
            }
            return key

    return None


# -------------------- main compare engine --------------------
def bfs_compare(tree, enable_logging=True):

    comparisons = {}
    stats = {
        "total_comparisons": 0,
        "successful_comparisons": 0,
        "failed_comparisons": 0,
        "sibling_comparisons": 0,
        "parent_child_comparisons": 0,
        "sibling_success": 0,
        "parent_child_success": 0
    }

    nodes = tree.get("nodes", {})

    # ---------- Build depth levels for sibling compares ----------
    levels = {}
    for nid, node in nodes.items():
        d = node.get("d")
        if d is not None:
            levels.setdefault(d, []).append(node)

    # ---------- Part 1: sibling comparisons ----------
    for d, nl in levels.items():
        arr = sorted(nl, key=lambda x: x["node_id"])
        total = len(arr) * (len(arr) - 1) // 2

        for i in tqdm(range(len(arr)), desc=f"Sibling comparisons (d={d})", leave=False):
            for j in range(i + 1, len(arr)):
                a = arr[i]
                b = arr[j]

                stats["total_comparisons"] += 1
                stats["sibling_comparisons"] += 1

                if enable_logging:
                    print(f"[Sibling Compare] {a['node_id']} <-> {b['node_id']}")

                key = maybe_add_comparison(a, b, comparisons)

                if key:
                    stats["successful_comparisons"] += 1
                    stats["sibling_success"] += 1
                else:
                    stats["failed_comparisons"] += 1

    # ---------- Part 2: parent-child comparisons ----------
    parent_child_pairs = []
    for pid, parent in nodes.items():
        for cid in parent.get("children_ids", []):
            if cid in nodes:
                parent_child_pairs.append((parent, nodes[cid]))

    for parent, child in tqdm(parent_child_pairs,
                              desc="Parent-child comparisons",
                              leave=False):

        stats["total_comparisons"] += 1
        stats["parent_child_comparisons"] += 1

        if enable_logging:
            print(f"[Parent-Child Compare] {parent['node_id']} -> {child['node_id']}")

        key = maybe_add_comparison(parent, child, comparisons)

        if key:
            stats["successful_comparisons"] += 1
            stats["parent_child_success"] += 1
        else:
            stats["failed_comparisons"] += 1

    return comparisons, stats



import argparse

# ============================================================
#                ***  Batch Processing Version  ***
# ============================================================

def main_batch():
    parser = argparse.ArgumentParser(description="Generate comparison pairs from revision trees")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing tree JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for comparison files")
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir

    print("\n=== 批量处理开始 ===")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")

    # ---------- 直接使用输出目录，不创建时间戳子目录 ----------
    run_dir = OUTPUT_DIR
    os.makedirs(run_dir, exist_ok=True)

    print("本次输出目录：", run_dir)

    # ---------- 扫描所有树文件 ----------
    # Recursively find all json files if needed, or just top level?
    # The previous code only looked at top level. Let's stick to that for now, 
    # but maybe the user's output structure has subfolders (part0, part1...).
    # The user's run_parallel_train.sh outputs to part0, part1, etc.
    # So we might need to handle that. 
    # For now, let's assume the user points to a specific part folder, or we can walk the directory.
    
    tree_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for f in files:
            if f.endswith(".json") and not f.startswith("config"): # Avoid config.json
                 tree_files.append(os.path.join(root, f))

    print(f"Found {len(tree_files)} tree files.")

    for in_path in tqdm(tree_files, desc="Processing trees"):
        try:
            tree_raw = load_json(in_path)
        except Exception as e:
            print(f"[ERROR] Failed to load {in_path}: {e}")
            continue
        
        # ---- 关键修复：兼容 list / dict 两种格式 ----
        if isinstance(tree_raw, list):
            if len(tree_raw) == 0:
                print(f"[WARN] empty tree list: {in_path}")
                continue
            tree = tree_raw[0]
        else:
            tree = tree_raw
        
        comparisons, stats = bfs_compare(tree, enable_logging=False)


        # ---- 构造输出 JSON（保留 metadata） ----
        out_json = {
            "question_id": tree.get("question_id"),
            "db_id": tree.get("db_id"),
            "question": tree.get("question"),
            "evidence": tree.get("evidence"),
            "difficulty": tree.get("difficulty"),
            "SQL": tree.get("SQL"),

            "comparisons": comparisons,
            "stats": stats
        }

        # Use relative path to maintain structure or just flat?
        # Let's flatten it for now as the original code did, but use the filename.
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(run_dir, f"compare_{base}.json")
        save_json(out_json, out_path)

    print("\n=== 批量处理结束 ===")
    print("结果全部保存在：", run_dir)



if __name__ == "__main__":
    main_batch()
