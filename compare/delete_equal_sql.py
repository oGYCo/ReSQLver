import json
import os
from tqdm import tqdm


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def filter_same_pairs(data):
    """
    删除 chosen == rejected 的样本
    """
    filtered = []
    removed = 0

    for item in data:
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        if chosen == rejected:
            removed += 1
            continue

        filtered.append(item)

    return filtered, removed


import argparse

def main():
    parser = argparse.ArgumentParser(description="Filter out same chosen/rejected pairs")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing DPO JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cleaned DPO JSON files")
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    total_before = 0
    total_after = 0
    total_removed = 0

    for fname in tqdm(files, desc="Filtering DPO files"):
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)

        try:
            data = load_json(in_path)
        except Exception as e:
            print(f"[ERROR] Cannot load {fname}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[SKIP] {fname} is not a list")
            continue

        filtered, removed = filter_same_pairs(data)

        save_json(filtered, out_path)

        total_before += len(data)
        total_after += len(filtered)
        total_removed += removed

    print("\n========== SUMMARY ==========")
    print(f"Total samples before: {total_before}")
    print(f"Total samples after : {total_after}")
    print(f"Total removed       : {total_removed}")
    print(f"Output directory    : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
