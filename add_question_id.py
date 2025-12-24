import json
import os

file_path = '/data/train/train.json'

def add_question_ids(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Error: The content of {path} is not a list.")
            return

        print(f"Processing {len(data)} items...")
        
        for idx, item in enumerate(data):
            # Insert question_id at the beginning if possible, or just add it
            # To make it look nice, we can reconstruct the dict, but simple assignment is fine.
            # If we want it to be the first key, we can do this:
            new_item = {"question_id": idx}
            new_item.update(item)
            data[idx] = new_item

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        print(f"Successfully added question_id to {len(data)} items in {path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    add_question_ids(file_path)
