import argparse
from pathlib import Path

from llm import LLMClassifier
from prompts import PROMPTS
from utils import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", default="zero_shot_v2")
args = parser.parse_args()

dev = load_jsonl(Path("data/dev.jsonl"))
effort = "low" if "gpt-oss" in args.model else None
clf = LLMClassifier(args.model, reasoning_effort=effort)

for row in dev[:5]:
    pred = clf.predict(PROMPTS[args.prompt](row["text"]))
    mark = "✓" if pred.label == row["label"] else "✗"
    print(f"{mark} จริง={row['label']:<4} ทาย={pred.label:<7} {pred.latency:.2f}s  out={pred.output_tokens}")
    print(f"   {row['text'][:80]}")
    print(f"   raw={pred.raw[:300]!r}\n")