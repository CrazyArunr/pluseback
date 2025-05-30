"""
Main entry point for the Pulse QA API
Combines both mongo_api and pos_api into a single FastAPI application
"""

import os
import sys

# Add the current directory to Python path for Azure App Service
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import using relative imports
from api.mongo_api import router as mongo_router
from api.pos_api import router as pos_router
from config import settings

# Create the main FastAPI app
app = FastAPI(
    title="Pulse QA Automation API",
    description="Combined API for test automation and analysis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Include routers without prefixes
app.include_router(mongo_router, tags=["MongoDB API"])
app.include_router(pos_router, tags=["POS API"])

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": ["mongodb", "gemini", "pos"],
        "version": "2.0.0"
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.DEFAULT_HOST,
        port=settings.DEFAULT_PORT
    ) 
