"""Check that the language model answers.  Run: python src/llm_check.py"""
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
model = init_chat_model(os.environ["LLM_MODEL"], temperature=0)
print(model.invoke("Reply with exactly one word: OK").text)