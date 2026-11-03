import argparse
import json
from pathlib import Path

from llm import LLMClassifier
from metrics import compute_metrics, print_report
from prompts import PROMPTS
from utils import load_jsonl

PRICES = {  # USD ต่อ 1M token: (input, output) ณ มิ.ย. 2026
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.11, 0.34),
    "qwen/qwen3.8-27b": (0.80, 4.00),
}


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[round(p / 100 * (len(ordered) - 1))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--prompt", default="zero_shot")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()

    rows = load_jsonl(Path(f"data/{args.split}.jsonl"))
    model_slug = args.model.replace("/", "_")
    out_path = Path(f"results/{args.split}/{model_slug}__{args.prompt}__run{args.run}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = {r["id"]: r for r in load_jsonl(out_path)} if out_path.exists() else {}
    print(f"มีผลเดิมแล้ว {len(done)}/{len(rows)} ข้อ")

    effort = "low" if "gpt-oss" in args.model else None
    clf = LLMClassifier(args.model, reasoning_effort=effort)
    build_messages = PROMPTS[args.prompt]

    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            pred = clf.predict(build_messages(row["text"]))
            record = {
                "id": row["id"],
                "label": row["label"],
                "pred": pred.label,
                "raw": pred.raw,
                "latency": pred.latency,
                "input_tokens": pred.input_tokens,
                "output_tokens": pred.output_tokens,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            done[row["id"]] = record
            print(f"\r{i}/{len(rows)}", end="", flush=True)
    print()

    results = [done[row["id"]] for row in rows]
    y_true = [r["label"] for r in results]
    y_pred = [r["pred"] for r in results]
    latencies = [r["latency"] for r in results]
    if args.model in PRICES:
        price_in, price_out = PRICES[args.model]
        cost_per_1k = sum(
            r["input_tokens"] * price_in + r["output_tokens"] * price_out for r in results
        ) / 1_000_000 / len(results) * 1000
        cost_text = f"${cost_per_1k:.4f} ต่อ 1,000 ข้อความ"
    else:
        cost_text = "ยังไม่ได้ใส่ราคาใน PRICES"
    avg_out = sum(r["output_tokens"] for r in results) / len(results)

    print(f"\n===== {args.model} | {args.prompt} | {args.split} | run {args.run} =====")
    print(compute_metrics(y_true, y_pred))
    print(f"latency p50={percentile(latencies, 50):.2f}s  p95={percentile(latencies, 95):.2f}s")
    print(f"ต้นทุน {cost_text} | output เฉลี่ย {avg_out:.0f} token")
    print_report(y_true, y_pred)


if __name__ == "__main__":
    main()