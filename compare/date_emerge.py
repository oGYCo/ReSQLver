import json
import os
from tqdm import tqdm


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge all JSON files into one")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing JSON files")
    parser.add_argument("--output_file", type=str, required=True, help="Output merged JSON file path")
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    OUTPUT_FILE = args.output_file

    all_data = []
    file_count = 0

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    for fname in tqdm(files, desc="Merging JSON files"):
        path = os.path.join(INPUT_DIR, fname)

        try:
            data = load_json(path)
        except Exception as e:
            print(f"[ERROR] Failed to load {fname}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[SKIP] {fname} is not a list, skipped")
            continue

        all_data.extend(data)
        file_count += 1

    save_json(all_data, OUTPUT_FILE)

    print("\n========== SUMMARY ==========")
    print(f"Files merged   : {file_count}")
    print(f"Total entries  : {len(all_data)}")
    print(f"Output file    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
