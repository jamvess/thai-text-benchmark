import random
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from metrics import LABELS
from run_eval import PRICES, percentile
from utils import load_jsonl

N_BOOT = 2000

# (ชื่อที่จะแสดง, model id สำหรับดูราคา, split ที่รัน, ชื่อไฟล์)
ENTRIES = [
    ("WangchanBERTa fine-tune", None, "test_full", "wangchanberta__finetune__run1"),
    ("Qwen3.8-27B zero-shot v2", "qwen/qwen3.8-27b", "test_full", "qwen_qwen3.8-27b__zero_shot_v2__run1"),
    ("gpt-oss-120b zero-shot v2", "openai/gpt-oss-120b", "test", "openai_gpt-oss-120b__zero_shot_v2__run1"),
    ("gpt-oss-20b zero-shot v2", "openai/gpt-oss-20b", "test", "openai_gpt-oss-20b__zero_shot_v2__run1"),
    ("Typhoon 2.5 zero-shot v2", "typhoon-v2.5-30b-a3b-instruct", "test", "typhoon-v2.5-30b-a3b-instruct__zero_shot_v2__run1"),
]


def macro_f1(y_true, y_pred, labels=LABELS):
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def resample(n, rng):
    return [rng.randrange(n) for _ in range(n)]


def bootstrap_ci(y_true, y_pred):
    rng = random.Random(0)
    scores = sorted(
        macro_f1([y_true[i] for i in idx], [y_pred[i] for i in idx])
        for idx in (resample(len(y_true), rng) for _ in range(N_BOOT))
    )
    return scores[int(0.025 * N_BOOT)], scores[int(0.975 * N_BOOT)]


def paired_diff_ci(y_true, pred_a, pred_b):
    rng = random.Random(0)
    diffs = []
    for _ in range(N_BOOT):
        idx = resample(len(y_true), rng)
        t = [y_true[i] for i in idx]
        diffs.append(macro_f1(t, [pred_b[i] for i in idx]) - macro_f1(t, [pred_a[i] for i in idx]))
    diffs.sort()
    observed = macro_f1(y_true, pred_b) - macro_f1(y_true, pred_a)
    return observed, diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT)]

def summarize(name, model_id, results):
    y_true = [r["label"] for r in results]
    y_pred = [r["pred"] for r in results]
    low, high = bootstrap_ci(y_true, y_pred)
    f1_q = f1_score(y_true, y_pred, labels=["q"], average="macro", zero_division=0)

    if results[0].get("latency") is not None:
        lat = [r["latency"] * 1000 for r in results]
        latency = f"{percentile(lat, 50):.0f} / {percentile(lat, 95):.0f}"
    else:
        latency = "-"

    if model_id in PRICES:
        p_in, p_out = PRICES[model_id]
        cost = sum(r["input_tokens"] * p_in + r["output_tokens"] * p_out for r in results)
        cost = f"${cost / 1_000_000 / len(results) * 1000:.3f}"
    else:
        cost = "-"

    return (
        f"| {name} | {len(results)} | {macro_f1(y_true, y_pred):.3f} [{low:.3f}, {high:.3f}] "
        f"| {macro_f1(y_true, y_pred, ['pos', 'neu', 'neg']):.3f} | {f1_q:.3f} "
        f"| {accuracy_score(y_true, y_pred):.1%} | {latency} | {cost} |"
    )

def baselines(rows):
    rng = random.Random(42)
    majority = [{"label": r["label"], "pred": "neu", "latency": None} for r in rows]
    rand = [{"label": r["label"], "pred": rng.choice(LABELS), "latency": None} for r in rows]
    return [("Majority (neu)", None, majority), ("Random", None, rand)]

HEADER = (
    "| วิธี | n | macro-F1 [95% CI] | macro-F1 (3 หมวด) | F1 q | accuracy | latency p50 / p95 (ms) | $/1k |\n"
    "|---|---|---|---|---|---|---|---|"
)

def load_entries():
    loaded = {}
    for name, model_id, split, filename in ENTRIES:
        path = Path(f"results/{split}/{filename}.jsonl")
        if not path.exists():
            print(f"⚠️  ยังไม่มีไฟล์: {path}")
            continue
        loaded[name] = (model_id, {r["id"]: r for r in load_jsonl(path)})
    return loaded

def table(title, rows, loaded, names):
    ids = [r["id"] for r in rows]
    print(f"\n### {title}\n")
    print(HEADER)
    for name, model_id, results in baselines(rows):
        print(summarize(name, model_id, results))
    for name in names:
        if name not in loaded:
            continue
        model_id, by_id = loaded[name]
        missing = [i for i in ids if i not in by_id]
        if missing:
            print(f"| {name} | ⚠️ ยังขาด {len(missing)} ข้อ | | | | | | |")
            continue
        print(summarize(name, model_id, [by_id[i] for i in ids]))


def main():
    loaded = load_entries()
    all_names = [e[0] for e in ENTRIES]

    test_600 = load_jsonl(Path("data/test.jsonl"))
    table("ตาราง A: ทุกวิธีบน test 600 ข้อชุดเดียวกัน", test_600, loaded, all_names)

    test_full = load_jsonl(Path("data/test_full.jsonl"))
    pair = ["WangchanBERTa fine-tune", "Qwen3.8-27B zero-shot v2"]
    table("ตาราง B: test เต็มชุด", test_full, loaded, pair)

    if all(n in loaded for n in pair):
        ids = [r["id"] for r in test_full]
        (_, a), (_, b) = loaded[pair[1]], loaded[pair[0]]
        if all(i in a and i in b for i in ids):
            y_true = [a[i]["label"] for i in ids]
            obs, low, high = paired_diff_ci(y_true, [a[i]["pred"] for i in ids], [b[i]["pred"] for i in ids])
            print(f"\nWangchanBERTa − Qwen (test เต็มชุด): {obs:+.3f}  [95% CI {low:+.3f}, {high:+.3f}]")

if __name__ == "__main__":
    main()