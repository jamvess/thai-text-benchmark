import random
from pathlib import Path

from metrics import LABELS, compute_metrics, print_report
from utils import load_jsonl

dev = load_jsonl(Path("data/dev.jsonl"))
y_true = [row["label"] for row in dev]

# baseline 1: ตอบหมวดที่มีมากที่สุดใน train (neu) ทุกข้อ
majority = ["neu"] * len(dev)

# baseline 2: สุ่มหมวดมั่วๆ
rng = random.Random(42)
random_guess = [rng.choice(LABELS) for _ in dev]

for name, y_pred in [("majority (neu)", majority), ("random", random_guess)]:
    print(f"===== {name} =====")
    print(compute_metrics(y_true, y_pred))
    print_report(y_true, y_pred)