import argparse
import random
from collections import Counter
from pathlib import Path

from utils import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="results/dev/openai_gpt-oss-20b__zero_shot__run1.jsonl")
parser.add_argument("--true", default="neu")
parser.add_argument("--pred", default="pos")
parser.add_argument("--n", type=int, default=15)
args = parser.parse_args()

results = load_jsonl(Path(args.file))
texts = {row["id"]: row["text"] for row in load_jsonl(Path("data/dev.jsonl"))}

errors = [r for r in results if r["label"] != r["pred"]]
print(f"ผิดทั้งหมด {len(errors)} ข้อ แยกตามคู่ (จริง → ทาย):")
for (true, pred), count in Counter((r["label"], r["pred"]) for r in errors).most_common():
    print(f"  {true} → {pred}: {count}")

selected = [r for r in errors if r["label"] == args.true and r["pred"] == args.pred]
print(f"\n===== ตัวอย่าง {args.true} → {args.pred} =====")
for r in random.Random(0).sample(selected, min(args.n, len(selected))):
    print(f"\n[{r['id']}] {texts[r['id']]}")