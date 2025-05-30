# Pulse QA API

A unified API service for test scenario generation, analysis, and reporting in the Pulse QA ecosystem.

## Overview

The Pulse QA API combines two main services:
1. **POS API**: Handles test scenario generation and analysis
2. **MongoDB API**: Manages test reporting and comparison

## Project Structure

```
pulse_qa_api/
├── api/
│   ├── __init__.py
│   ├── mongo_api.py    # MongoDB reporting and analysis endpoints
│   └── pos_api.py      # Test scenario generation endpoints
├── config/
│   ├── __init__.py
│   └── settings.py     # Configuration settings
├── utils/
│   ├── __init__.py
│   └── helpers.py      # Common utility functions
├── __init__.py
├── main.py            # Main application entry point
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with the following variables:
   ```
   CUSTOM_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   MONGO_URI=your_mongodb_connection_string
   DB_NAME=your_database_name
   COLLECTION_NAME=your_collection_name
   ```

## Running the API

Start the server:
```bash
python -m uvicorn pulse_qa_api.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: `http://localhost:8000/docs`
- Alternative API documentation: `http://localhost:8000/redoc`

### Main Endpoints

#### POS API Endpoints
- `POST /pos/generate-scenario`: Generate test scenarios
- `POST /pos/chat`: Chat with the AI about test scenarios
- `POST /pos/generate-ai-data`: Generate AI test data
- `POST /pos/predict-next-step`: Predict next test steps

#### MongoDB API Endpoints
- `GET /mongo/compare-projects`: Compare two test projects
- `GET /mongo/project/failures-summary`: Get failure analysis
- `GET /mongo/compare-projects/ai`: AI-powered project comparison
- `POST /mongo/report-chat`: Chat about test reports

## Development

### Adding New Features
1. Create new endpoints in the appropriate API file (`pos_api.py` or `mongo_api.py`)
2. Add any shared utilities to `helpers.py`
3. Update configuration in `settings.py` if needed
4. Add new dependencies to `requirements.txt`

### Testing
Run tests (when implemented):
```bash
pytest
```

## License

This project is proprietary and confidential.

## Support

For support, please contact the Pulse QA team. 