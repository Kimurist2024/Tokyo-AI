from api_key_manager import APIKeyManager
from openai import OpenAI

manager = APIKeyManager()
client = OpenAI(api_key=manager.get_key("OPENAI_API_KEY"))

jobs = client.fine_tuning.jobs.list()
for job in jobs.data:
    print(f"ID: {job.id} | Status: {job.status} | Model: {job.model}")