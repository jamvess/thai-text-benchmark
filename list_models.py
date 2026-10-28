import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

for model in sorted(client.models.list().data, key=lambda m: m.id):
    print(f"{model.id:<50} owner={model.owned_by:<15} context={model.context_window}")