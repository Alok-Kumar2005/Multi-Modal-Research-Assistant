from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    gemini_model_name = "gemini-1.5-flash"
    gemini_model_kwargs = {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_output_tokens": 512,
        "top_k": 40,
    }
    gemini_api_key = os.getenv("GOOGLE_API_KEY")

    groq_model_name = "gemma2-9b-it"
    groq_model_kwargs = {
        "temperature": 0.2,
        # "top_p": 0.95,
        "max_tokens": 512
    }
    groq_api_key = os.getenv("GROQ_API_KEY")

    serper_api_key = os.getenv("SERPER_API_KEY")