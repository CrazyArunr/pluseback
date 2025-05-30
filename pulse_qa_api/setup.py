from setuptools import setup, find_packages

setup(
    name="pulse_qa_api",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        # Add other dependencies from your requirements.txt
    ],
    python_requires=">=3.8",
) 