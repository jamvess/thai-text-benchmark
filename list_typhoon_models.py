import os

import httpx
from dotenv import load_dotenv

load_dotenv()
response = httpx.get(
    "https://api.opentyphoon.ai/v1/models",
    headers={"Authorization": f"Bearer {os.environ['TYPHOON_API_KEY']}"},
)
print("status:", response.status_code)

data = response.json()
items = data if isinstance(data, list) else data.get("data", [])
for item in items:
    print(item.get("id") if isinstance(item, dict) else item)