"""
Configuration settings for Pulse QA API
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    MONGO_URI = os.getenv("MONGO_URI", "")
    DB_NAME = os.getenv("DB_NAME", "testDB")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "generate_reports")
    CONNECT_TIMEOUT_MS = int(os.getenv("CONNECT_TIMEOUT_MS", "7000"))
    SOCKET_TIMEOUT_MS = int(os.getenv("SOCKET_TIMEOUT_MS", "70000"))
    
    # CORS Configuration
    CORS_ORIGINS = ["http://localhost:5173", "*"]
    CORS_METHODS = ["*"]
    CORS_HEADERS = ["*"]
    
    # Server Configuration
    DEFAULT_PORT = int(os.getenv("DEFAULT_PORT", "8000"))
    DEFAULT_HOST = os.getenv("DEFAULT_HOST", "0.0.0.0")

settings = Settings()
