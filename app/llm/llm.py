# llm.py
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load .env variables
load_dotenv()

class LLMClient:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )

    def chat(self, prompt: str) -> str:
        """Simple wrapper to send a prompt and get response text"""
        response = self.llm.invoke(prompt)
        return response.content
