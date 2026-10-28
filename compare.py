import argparse
import random
from pathlib import Path

from sklearn.metrics import f1_score

from metrics import LABELS
from utils import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("file_a")
parser.add_argument("file_b")
parser.add_argument("--n-boot", type=int, default=2000)
args = parser.parse_args()

a = {r["id"]: r for r in load_jsonl(Path(args.file_a))}
b = {r["id"]: r for r in load_jsonl(Path(args.file_b))}
ids = sorted(a.keys() & b.keys())
y_true = [a[i]["label"] for i in ids]
pred_a = [a[i]["pred"] for i in ids]
pred_b = [b[i]["pred"] for i in ids]

only_a = sum(pa == t and pb != t for t, pa, pb in zip(y_true, pred_a, pred_b))
only_b = sum(pb == t and pa != t for t, pa, pb in zip(y_true, pred_a, pred_b))
print(f"ทั้งหมด {len(ids)} ข้อ | ถูกเฉพาะ A: {only_a} | ถูกเฉพาะ B: {only_b}")


def macro_f1(indices, preds):
    return f1_score(
        [y_true[i] for i in indices],
        [preds[i] for i in indices],
        labels=LABELS,
        average="macro",
        zero_division=0,
    )


n = len(ids)
observed = macro_f1(range(n), pred_b) - macro_f1(range(n), pred_a)

rng = random.Random(0)
diffs = []
for _ in range(args.n_boot):
    sample = [rng.randrange(n) for _ in range(n)]
    diffs.append(macro_f1(sample, pred_b) - macro_f1(sample, pred_a))
diffs.sort()
low = diffs[int(0.025 * len(diffs))]
high = diffs[int(0.975 * len(diffs))]

print(f"macro-F1 ของ B − A = {observed:+.3f}  (95% CI: {low:+.3f} ถึง {high:+.3f})")
print("ช่วงคร่อม 0 → ยังสรุปไม่ได้" if low < 0 < high else "ช่วงไม่คร่อม 0 → ความต่างน่าจะเป็นของจริง")