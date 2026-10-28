from pathlib import Path

from datasets import load_dataset

from utils import save_jsonl

# เรียงสลับหมวด pos, neu, neg, q วนสองรอบ
IDS = [1594, 1445, 10439, 13235, 13966, 2030, 11105, 7613]

train = load_dataset("pythainlp/wisesight_sentiment")["train"]
names = train.features["category"].names

rows = [
    {"id": f"train-{i}", "text": train[i]["texts"], "label": names[train[i]["category"]]}
    for i in IDS
]
save_jsonl(rows, Path("data/few_shot.jsonl"))

for row in rows:
    print(f"[{row['label']}] {row['text']}")