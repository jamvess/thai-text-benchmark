import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

from metrics import LABELS

load_dotenv()

PROVIDERS = {  # ชื่อ provider: (base_url, ชื่อตัวแปรใน .env)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "typhoon": ("https://api.opentyphoon.ai/v1", "TYPHOON_API_KEY"),
}


def provider_for(model: str) -> str:
    return "typhoon" if model.startswith("typhoon") else "groq"


@dataclass
class Prediction:
    raw: str
    label: str
    latency: float
    input_tokens: int
    output_tokens: int


def parse_label(raw: str) -> str:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    text = (match.group(1) if match else raw).strip().lower()
    return text if text in LABELS else "invalid"


def retry_after_seconds(error: RateLimitError) -> float | None:
    header = error.response.headers.get("retry-after")
    if header:
        try:
            return float(header) + 1
        except ValueError:
            pass
    match = re.search(r"again in (?:(\d+)m)?([\d.]+)s", str(error))
    if not match:
        return None
    return int(match.group(1) or 0) * 60 + float(match.group(2)) + 1


class LLMClassifier:
    def __init__(self, model: str, reasoning_effort: str | None = None):
        base_url, key_env = PROVIDERS[provider_for(model)]
        self.client = OpenAI(api_key=os.environ[key_env], base_url=base_url, max_retries=0)
        self.model = model
        self.reasoning_effort = reasoning_effort

    def predict(self, messages: list[dict]) -> Prediction:
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        for attempt in range(30):
            try:
                start = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=1000,
                    **extra,
                )
                latency = time.perf_counter() - start
                break
            except RateLimitError as e:
                wait = retry_after_seconds(e) or 2**attempt
                print(f"\n[rate limit] รอ {wait:.0f}s")
                time.sleep(wait)
            except (APIConnectionError, InternalServerError) as e:
                wait = min(2**attempt, 60)
                print(f"\n[{type(e).__name__}] รอ {wait}s แล้วลองใหม่")
                time.sleep(wait)
        else:
            raise RuntimeError("ลองครบ 30 ครั้งแล้วยังไม่สำเร็จ")

        raw = response.choices[0].message.content or ""
        return Prediction(
            raw=raw,
            label=parse_label(raw),
            latency=latency,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )