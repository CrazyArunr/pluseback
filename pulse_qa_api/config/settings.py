"""
Configuration settings for the Pulse QA API
"""

# API Configuration
CUSTOM_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1dWlkIjoiOGNhNjY5ODktNDA4OC00MzNmLTkyZjItYjkyMTVlM2Y4ZDUzIiwiZXhwIjoxNzQ5MDI4MDI3LCJvcmdfdXVpZCI6IjkxMjg1NWYzLTdmYjItNGFlMy1hZmM2LWYwZTMwZjcxNjViMSJ9.sxhb03ezpPjYUzJasIIZjK8-_QYM3RU3FfBIavZJqUQ"
CUSTOM_API_BASE = "https://http.llm.proxy.prod.s9t.link"
CUSTOM_MODEL = "llama3_1"
CUSTOM_HEADERS = {"id": "c22eebe5-e034-472d-8726-8fd7ebfa5325"}
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# MongoDB Configuration
MONGO_URI = "mongodb+srv://vik:1234@pulsecluster.nmaugii.mongodb.net/testDB?retryWrites=true&w=majority"
DB_NAME = "testDB"
COLLECTION_NAME = "generate_reports"
CONNECT_TIMEOUT_MS = 7000
SOCKET_TIMEOUT_MS = 70000

# Gemini AI Configuration
GEMINI_API_KEY = "AIzaSyB9RCFA0V6gbBUwK4WEp7qSDeDcm9rfhuY"

# CORS Configuration
CORS_ORIGINS = ["http://localhost:5173", "*"]
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# Server Configuration
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0" 