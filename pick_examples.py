import random
import re

from datasets import load_dataset

train = load_dataset("pythainlp/wisesight_sentiment")["train"]
names = train.features["category"].names
texts = train["texts"]
labels = [names[c] for c in train["category"]]
rng = random.Random(7)

QUESTION = re.compile(r"ไหม|มั้ย|มั๊ย|หรอ|เหรอ|อะไร|ยังไง|อย่างไร|ที่ไหน|เท่าไร|เท่าไหร่|\?")
PROMO = re.compile(r"แจก|โปร|ลดราคา|ส่วนลด|ร่วมสนุก|สนใจ|inbox|สั่งซื้อ")

buckets = {
    "q (คำถามจริง)": lambda t, l: l == "q",
    "neu ที่เป็นประโยคคำถาม": lambda t, l: l == "neu" and QUESTION.search(t),
    "neu ที่ดูเหมือนโฆษณา": lambda t, l: l == "neu" and PROMO.search(t),
    "pos": lambda t, l: l == "pos",
    "neg ที่เป็นประโยคคำถาม (อาจประชด)": lambda t, l: l == "neg" and QUESTION.search(t),
    "neg": lambda t, l: l == "neg",
}

for title, condition in buckets.items():
    indices = [i for i, (t, l) in enumerate(zip(texts, labels)) if condition(t, l) and len(t) < 200]
    print(f"\n===== {title} (มีทั้งหมด {len(indices)} ข้อ) =====")
    for i in rng.sample(indices, min(8, len(indices))):
        print(f"[train-{i}] {texts[i]}")