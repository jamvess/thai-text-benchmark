# Final evaluation config (ล็อก 28 ต.ค. 2026 ก่อนรัน test)

| โมเดล | prompt / วิธี | split |
|---|---|---|
| qwen/qwen3.8-27b | zero_shot_v2 | test_full (2,671) |
| wangchanberta-base-att-spm-uncased | fine-tune 3 epochs, lr 3e-5 | test_full (2,671) |
| openai/gpt-oss-120b | zero_shot_v2, reasoning_effort=low | test (600) |
| openai/gpt-oss-20b | zero_shot_v2, reasoning_effort=low | test (600) |
| typhoon-v2.5-30b-a3b-instruct | zero_shot_v2 | test (600) |
| baselines (majority, random) | - | ทั้งสอง split |

- เกณฑ์หลัก: macro-F1 (4 หมวด) พร้อม 95% bootstrap CI
- เกณฑ์รอง: macro-F1 3 หมวด (pos/neu/neg), F1 ของแต่ละหมวด, latency p50/p95, ต้นทุนต่อ 1,000 ข้อความ
- ไม่ได้ใช้ few-shot เพราะบน dev ไม่ได้ช่วยแบบมีนัยสำคัญ (CI −0.020 ถึง +0.044) และใช้ input มากขึ้นราว 2 เท่า
- temperature = 0 ทุกโมเดล รันโมเดลละ 1 รอบ