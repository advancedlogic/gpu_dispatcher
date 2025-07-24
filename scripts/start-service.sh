#!/bin/bash
# GPU Stats Server startup script for systemd

# Change to the project directory
cd /home/udg/projects/tt/gpu_stats

# Activate virtual environment and start the server
source .venv/bin/activate
exec uvicorn gpu_server:app --env-file .env
