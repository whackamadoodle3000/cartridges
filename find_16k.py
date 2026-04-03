import json
from pathlib import Path

JSONL_FILE = "qasper_e.jsonl"
TARGET_LINE = 203
OUTPUT_FILE = "qasper_e_line_203_context.txt"

def main():
    jsonl_path = Path(JSONL_FILE)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx == TARGET_LINE:
                obj = json.loads(line)
                context = obj.get("context", "")

                if not context:
                    print(f"No context found on line {TARGET_LINE}")
                    return

                with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
                    out.write(context)

                print(f"Saved context from line {TARGET_LINE} to: {OUTPUT_FILE}")
                print(f"Question: {obj.get('input', '')}")
                print(f"Dataset: {obj.get('dataset', '')}")
                print(f"ID: {obj.get('_id', '')}")
                return

    print(f"Line {TARGET_LINE} not found in {JSONL_FILE}")

if __name__ == "__main__":
    main()