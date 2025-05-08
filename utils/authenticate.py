from google import genai
from dotenv import load_dotenv
import os


def authenticate_gemini():
    load_dotenv()
    gemini_api_key = os.getenv("gemini_api_key")
    client = genai.Client(api_key=gemini_api_key)
    print("Gemini API client authenticated successfully.")
    return client