import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("List of models:\n")
try:
    for m in client.models.list():
        print(f"Model: {m.name}")
        # print(f"Supported methods: {m.supported_generation_methods if hasattr(m, 'supported_generation_methods') else 'unknown'}")
except Exception as e:
    print(f"Error listing models: {e}")

print("List of models that support embedContent:\n")
for m in client.models.list():
    for action in m.supported_actions:
        if action == "embedContent":
            print(m.name)