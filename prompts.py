from functools import lru_cache
from pathlib import Path

from utils import load_jsonl

SYSTEM = """คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์ความรู้สึก (sentiment) ของข้อความภาษาไทยจากโซเชียลมีเดีย
ผลการจัดหมวดจะใช้วัดความรู้สึกของลูกค้าที่มีต่อสินค้าและบริการ"""

LABEL_GUIDE = """<labels>
pos: แสดงความรู้สึกด้านบวก เช่น ชื่นชม พอใจ แนะนำ
neu: ไม่แสดงความรู้สึกชัดเจน เช่น ให้ข้อมูล ข่าว โฆษณา
neg: แสดงความรู้สึกด้านลบ เช่น บ่น ไม่พอใจ ผิดหวัง
q: เป็นคำถาม หรือสอบถามข้อมูล
</labels>"""


def escape(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def zero_shot(text: str) -> list[dict]:
    user = f"""{LABEL_GUIDE}

<text>
{escape(text)}
</text>

จัดหมวดข้อความใน <text> เป็นหนึ่งใน pos, neu, neg, q
ตอบเป็นชื่อหมวดคำเดียว ตัวพิมพ์เล็ก ไม่ต้องอธิบาย"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


LABEL_GUIDE_V2 = """<labels>
pos: ผู้เขียนแสดงความรู้สึกด้านบวกของตัวเอง เช่น ชื่นชม พอใจ ประทับใจ หรือแสดงความสนใจอยากได้สินค้า
neu: ผู้เขียนไม่ได้แสดงความรู้สึกของตัวเองชัดเจน รวมถึงข้อความต่อไปนี้ แม้จะมีคำเชิงบวกหรือลบอยู่ด้วย
  - โฆษณา โปรโมชัน กิจกรรมแจกของ หรือข้อความที่เขียนในมุมของผู้ขายหรือแบรนด์
  - ข่าว การให้ข้อมูล การเล่าเหตุการณ์ทั่วไป
  - คำถามลอยๆ ที่ไม่ได้ต้องการคำตอบจริง หรือการชวนคุยกับเพื่อน
neg: ผู้เขียนแสดงความรู้สึกด้านลบ เช่น บ่น ไม่พอใจ ผิดหวัง ประชดประชัน หรือบอกว่าสินค้าหรือบริการเจ้าอื่นดีกว่า
q: ผู้เขียนตั้งใจถามเพื่อขอข้อมูลเกี่ยวกับสินค้าหรือบริการจริงๆ และคาดหวังว่าจะได้คำตอบ
</labels>

<rules>
- ตัดสินจากความรู้สึกของผู้เขียนเอง ไม่ใช่จากคำเชิงบวกหรือเชิงลบที่ปรากฏในข้อความ
- ถ้าเป็นคำถามเชิงประชดประชัน ให้เป็น neg ไม่ใช่ q
</rules>"""


def zero_shot_v2(text: str) -> list[dict]:
    user = f"""{LABEL_GUIDE_V2}

<text>
{escape(text)}
</text>

จัดหมวดข้อความใน <text> เป็นหนึ่งใน pos, neu, neg, q ตามนิยามใน <labels> และกติกาใน <rules>
ตอบเป็นชื่อหมวดคำเดียว ตัวพิมพ์เล็ก ไม่ต้องอธิบาย"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


@lru_cache
def few_shot_examples() -> str:
    rows = load_jsonl(Path("data/few_shot.jsonl"))
    return "\n".join(
        f"<example>\n<message>{escape(r['text'])}</message>\n<label>{r['label']}</label>\n</example>"
        for r in rows
    )


def few_shot(text: str) -> list[dict]:
    user = f"""{LABEL_GUIDE_V2}

<examples>
{few_shot_examples()}
</examples>

<text>
{escape(text)}
</text>

จัดหมวดข้อความใน <text> เป็นหนึ่งใน pos, neu, neg, q ตามนิยามใน <labels> กติกาใน <rules> และแนวทางเดียวกับตัวอย่างใน <examples>
ตอบเป็นชื่อหมวดคำเดียว ตัวพิมพ์เล็ก ไม่ต้องอธิบาย"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]

PROMPTS = {
    "zero_shot": zero_shot,
    "zero_shot_v2": zero_shot_v2,
    "few_shot": few_shot,
}