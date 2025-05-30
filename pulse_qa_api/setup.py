from setuptools import setup, find_packages

setup(
    name="pulse_qa_api",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0,<0.69.0",
        "uvicorn>=0.15.0,<0.16.0",
        "python-multipart>=0.0.5,<0.1.0",
        "python-dotenv>=0.19.0,<0.20.0",
        "pymongo>=4.0.0,<5.0.0",
        "openai>=1.0.0,<2.0.0",
        "langchain>=0.1.0,<0.2.0",
        "langchain-openai>=0.0.2,<0.1.0",
        "google-generativeai>=0.3.0,<0.4.0",
        "sentence-transformers>=2.2.0,<3.0.0",
        "faiss-cpu>=1.7.0,<2.0.0",
        "pandas>=1.5.0,<2.0.0",
        "requests>=2.28.0,<3.0.0",
        "pydantic",
        "python-jose[cryptography]>=3.3.0,<4.0.0",
        "passlib[bcrypt]>=1.7.4,<2.0.0",
        "aiofiles>=0.8.0,<0.9.0"
    ],
    python_requires=">=3.8",
    author="Pulse QA Team",
    description="A unified API service for test scenario generation, analysis, and reporting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
) 
