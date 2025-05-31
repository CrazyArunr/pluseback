#!/bin/bash

# Install the package in development mode
cd /opt/render/project/src
pip install -e .

# Add the application directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/opt/render/project/src

# Start Gunicorn
gunicorn --bind=0.0.0.0:8000 --workers=4 --worker-class=uvicorn.workers.UvicornWorker pulse_qa_api.main:app 