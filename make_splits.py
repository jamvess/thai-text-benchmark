import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42
DATA_DIR = Path("data")


def stratified_sample(split, split_name, n):
    labels = split["category"]
    indices = list(range(len(split)))
    picked, _ = train_test_split(indices, train_size=n, stratify=labels, random_state=SEED)
    names = split.features["category"].names
    return [
        {"id": f"{split_name}-{i}", "text": split[i]["texts"], "label": names[split[i]["category"]]}
        for i in sorted(picked)
    ]


def save_jsonl(rows, path):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


ds = load_dataset("pythainlp/wisesight_sentiment")
DATA_DIR.mkdir(exist_ok=True)

dev = stratified_sample(ds["validation"], "validation", 300)
test = stratified_sample(ds["test"], "test", 600)
save_jsonl(dev, DATA_DIR / "dev.jsonl")
save_jsonl(test, DATA_DIR / "test.jsonl")

for name, rows in [("dev", dev), ("test", test)]:
    counts = Counter(r["label"] for r in rows)
    print(name, len(rows), dict(counts))

print("\nตัวอย่างจาก dev:")
for row in dev[:10]:
    print(f"[{row['label']}] {row['text'][:100]}")

full_test = [
    {"id": f"test-{i}", "text": t, "label": ds["test"].features["category"].names[c]}
    for i, (t, c) in enumerate(zip(ds["test"]["texts"], ds["test"]["category"]))
]
save_jsonl(full_test, DATA_DIR / "test_full.jsonl")