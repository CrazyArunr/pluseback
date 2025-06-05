import os
from pydantic_settings import BaseSettings
from typing import List, Dict

class Settings(BaseSettings):
    # API Configuration
    CUSTOM_API_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1dWlkIjoiOGNhNjY5ODktNDA4OC00MzNmLTkyZjItYjkyMTVlM2Y4ZDUzIiwiZXhwIjoxNzQ5MDI4MDI3LCJvcmdfdXVpZCI6IjkxMjg1NWYzLTdmYjItNGFlMy1hZmM2LWYwZTMwZjcxNjViMSJ9.sxhb03ezpPjYUzJasIIZjK8-_QYM3RU3FfBIavZJqUQ"
    CUSTOM_API_BASE: str = "https://http.llm.proxy.prod.s9t.link"
    CUSTOM_MODEL: str = "llama3_1"
    CUSTOM_HEADERS: Dict[str, str] = {"id": "c22eebe5-e034-472d-8726-8fd7ebfa5325"}
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

    # MongoDB Configuration
    MONGO_URI: str = "mongodb+srv://vik:1234@pulsecluster.nmaugii.mongodb.net/testDB?retryWrites=true&w=majority"
    DB_NAME: str = "testDB"
    COLLECTION_NAME: str = "generate_reports"
    CONNECT_TIMEOUT_MS: int = 7000
    SOCKET_TIMEOUT_MS: int = 70000

    # Gemini AI Configuration
    GEMINI_API_KEY: str = "AIzaSyB9RCFA0V6gbBUwK4WEp7qSDeDcm9rfhuY"

    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "*"]
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]

    # Server Configuration
    DEFAULT_PORT: int = 8000
    DEFAULT_HOST: str = "0.0.0.0"

settings = Settings()
