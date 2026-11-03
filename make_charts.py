from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from final_report import baselines, bootstrap_ci, load_entries, macro_f1
from run_eval import percentile
from utils import load_jsonl

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "fine-tuned": "#2a78d6", "LLM API": "#eb6834", "baseline": "#898781",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7",
        "grid": "#2c2c2a", "axis": "#383835",
        "fine-tuned": "#3987e5", "LLM API": "#d95926", "baseline": "#898781",
    },
}
GROUPS = ["fine-tuned", "LLM API", "baseline"]

# ชื่อใน final_report.py → (ชื่อสั้นในกราฟ, กลุ่ม)
SHORT_NAMES = {
    "WangchanBERTa fine-tune": ("WangchanBERTa fine-tune", "fine-tuned"),
    "Qwen3.8-27B zero-shot v2": ("Qwen3.8-27B", "LLM API"),
    "gpt-oss-120b zero-shot v2": ("gpt-oss-120b", "LLM API"),
    "gpt-oss-20b zero-shot v2": ("gpt-oss-20b", "LLM API"),
    "Typhoon 2.5 zero-shot v2": ("Typhoon 2.5", "LLM API"),
}

# ตำแหน่งป้ายชื่อในกราฟ latency: (เลื่อนแนวนอน, เลื่อนแนวตั้ง เป็น point, การจัดวาง)
LABEL_OFFSETS = {
    "Qwen3.8-27B": (-9, 0, "right"),
    "Typhoon 2.5": (-9, -12, "right"),
    "gpt-oss-20b": (9, -14, "left"),
}


def make_row(name, group, results):
    y_true = [r["label"] for r in results]
    y_pred = [r["pred"] for r in results]
    low, high = bootstrap_ci(y_true, y_pred)
    latencies = [r["latency"] * 1000 for r in results if r.get("latency") is not None]
    return {
        "name": name,
        "group": group,
        "f1": macro_f1(y_true, y_pred),
        "low": low,
        "high": high,
        "p50_ms": percentile(latencies, 50) if latencies else None,
    }


def collect():
    test_600 = load_jsonl(Path("data/test.jsonl"))
    ids = [r["id"] for r in test_600]
    rows = []
    for full_name, (_, by_id) in load_entries().items():
        name, group = SHORT_NAMES[full_name]
        rows.append(make_row(name, group, [by_id[i] for i in ids]))
    for name, _, results in baselines(test_600):
        rows.append(make_row(name, "baseline", results))
    return rows


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], labelsize=10, length=0)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])


def legend(ax, t, groups, loc):
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=t[g], label=g)
        for g in groups
    ]
    leg = ax.legend(handles=handles, frameon=False, fontsize=10, loc=loc)
    for text in leg.get_texts():
        text.set_color(t["ink2"])


def chart_f1(rows, theme, path):
    t = THEMES[theme]
    rows = sorted(rows, key=lambda r: r["f1"])
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    style_axes(ax, t)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

    for y, r in enumerate(rows):
        color = t[r["group"]]
        ax.plot([r["low"], r["high"]], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(r["f1"], y, "o", markersize=8, color=color,
                markeredgecolor=t["surface"], markeredgewidth=2)
        ax.text(r["high"] + 0.012, y, f"{r['f1']:.3f}", va="center", fontsize=9, color=t["ink2"])

    ax.set_yticks(range(len(rows)), [r["name"] for r in rows])
    ax.tick_params(axis="y", labelcolor=t["ink"])
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("macro-F1 (dot) with 95% bootstrap CI (line)")
    ax.set_title("macro-F1 on the same 600 test messages", loc="left",
                 fontsize=12, color=t["ink"], pad=12)
    legend(ax, t, [g for g in GROUPS if any(r["group"] == g for r in rows)], "lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def chart_latency(rows, theme, path):
    t = THEMES[theme]
    rows = [r for r in rows if r["p50_ms"]]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    style_axes(ax, t)
    ax.grid(color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xscale("log")

    for r in rows:
        color = t[r["group"]]
        ax.plot([r["p50_ms"]] * 2, [r["low"], r["high"]], color=color, linewidth=2,
                solid_capstyle="round")
        ax.plot(r["p50_ms"], r["f1"], "o", markersize=8, color=color,
                markeredgecolor=t["surface"], markeredgewidth=2)
        dx, dy, ha = LABEL_OFFSETS.get(r["name"], (9, 0, "left"))
        ax.annotate(r["name"], (r["p50_ms"], r["f1"]), xytext=(dx, dy), ha=ha,
                    textcoords="offset points", va="center", fontsize=9, color=t["ink"])

    ax.set_xlim(5, 4000)
    ax.set_ylim(0.4, 0.8)
    ax.set_xticks([10, 30, 100, 300, 1000], ["10", "30", "100", "300", "1,000"])
    ax.set_xlabel("latency p50 per message (ms, log scale)")
    ax.set_ylabel("macro-F1")
    ax.set_title("Accuracy vs. speed: higher and further left is better", loc="left",
                 fontsize=12, color=t["ink"], pad=12)
    legend(ax, t, [g for g in GROUPS if any(r["group"] == g for r in rows)], "upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    out = Path("images")
    out.mkdir(exist_ok=True)
    rows = collect()
    for theme in THEMES:
        chart_f1(rows, theme, out / f"f1_{theme}.png")
        chart_latency(rows, theme, out / f"latency_{theme}.png")
    print("บันทึกรูป 4 ไฟล์ไว้ใน images/ แล้ว")