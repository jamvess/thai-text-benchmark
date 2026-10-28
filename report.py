import argparse
from pathlib import Path

from metrics import compute_metrics, print_report
from run_eval import percentile
from utils import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()

results = load_jsonl(Path(args.file))
y_true = [r["label"] for r in results]
y_pred = [r["pred"] for r in results]
latencies = [r["latency"] for r in results]

print(f"===== {args.file} =====")
print(compute_metrics(y_true, y_pred))
print(f"latency p50={percentile(latencies, 50) * 1000:.1f}ms  p95={percentile(latencies, 95) * 1000:.1f}ms")
print_report(y_true, y_pred)