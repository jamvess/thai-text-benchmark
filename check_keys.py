import os

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI

from llm import PROVIDERS

load_dotenv()

tests = {"groq": "qwen/qwen3.8-27b", "typhoon": "typhoon-v2.5-30b-a3b-instruct"}

for name, model in tests.items():
    base_url, key_env = PROVIDERS[name]
    key = os.environ.get(key_env, "")
    cleaned = key.strip().strip('"').strip("'")
    print(f"{name}: มี key={bool(key)} | ยาว {len(key)} ตัวอักษร | มีช่องว่างหรือเครื่องหมายคำพูดเกิน={key != cleaned}")

    client = OpenAI(api_key=key, base_url=base_url, max_retries=0)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ตอบคำว่า ok"}],
            max_tokens=5,
        )
        print(f"  ✓ ใช้ได้: {response.choices[0].message.content!r}")
    except AuthenticationError as e:
        print(f"  ✗ 401: {str(e)[:200]}")