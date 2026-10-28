from collections import Counter

from datasets import load_dataset

ds = load_dataset("pythainlp/wisesight_sentiment")
print(ds)
print(ds["train"].features)

label_names = ds["train"].features["category"].names
print("\nlabels:", label_names)

for split in ["train", "validation", "test"]:
    counts = Counter(ds[split]["category"])
    total = len(ds[split])
    dist = "  ".join(f"{label_names[i]} {counts[i] / total:.1%}" for i in range(len(label_names)))
    print(f"{split:<10} {total:>6}  {dist}")

print("\nตัวอย่างหมวดละ 3 ข้อความ:")
train = ds["train"]
for label_id, name in enumerate(label_names):
    examples = [row["texts"] for row in train if row["category"] == label_id][:3]
    print(f"\n[{name}]")
    for text in examples:
        print("  -", text[:120].replace("\n", " "))