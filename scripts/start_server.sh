#!/bin/bash

# GPU Stats Server Startup Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting GPU Statistics Server${NC}"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 is not installed or not in PATH${NC}"
    exit 1
fi

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ nvidia-smi found${NC}"
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    echo -e "${BLUE}📊 Detected $GPU_COUNT GPU(s)${NC}"
else
    echo -e "${YELLOW}⚠️  nvidia-smi not found - GPU monitoring may not work${NC}"
fi

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo -e "${GREEN}✅ Virtual environment found${NC}"
    PYTHON_CMD=".venv/bin/python"
else
    echo -e "${YELLOW}⚠️  No virtual environment found, using system Python${NC}"
    PYTHON_CMD="python3"
fi

echo -e "${BLUE}🔍 Checking dependencies...${NC}"
if $PYTHON_CMD -c "import fastapi, uvicorn, dotenv" 2>/dev/null; then
    echo -e "${GREEN}✅ FastAPI, Uvicorn, and python-dotenv are installed${NC}"
else
    echo -e "${RED}❌ FastAPI, Uvicorn, or python-dotenv is not installed${NC}"
    echo -e "${YELLOW}📦 Installing requirements...${NC}"
    $PYTHON_CMD -m pip install -r requirements.txt
fi

# Check for .env file
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env configuration file found${NC}"
else
    echo -e "${YELLOW}⚠️  No .env file found, using default configuration${NC}"
    echo -e "${BLUE}💡 You can copy .env.example to .env to customize settings${NC}"
fi

# Start the server
echo -e "${BLUE}🌐 Starting FastAPI server with environment configuration${NC}"
echo -e "${BLUE}📚 API Documentation available at http://localhost:8000/docs${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Run the server
$PYTHON_CMD gpu_server.py
