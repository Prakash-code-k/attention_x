#!/bin/bash
# ─── AttentionX Quick Start Script ───────────────────────────────────────────
# Usage: chmod +x start.sh && ./start.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "  ╔═══════════════════════════════════════╗"
echo "  ║  ⚡ AttentionX – Content Repurposer  ║"
echo "  ╚═══════════════════════════════════════╝"
echo -e "${NC}"

# ─── Check prerequisites ──────────────────────────────────────────────────────
echo -e "${YELLOW}Checking prerequisites...${NC}"

command -v python3 >/dev/null 2>&1 || { echo -e "${RED}❌ Python 3 required${NC}"; exit 1; }
echo -e "${GREEN}✅ Python 3 found: $(python3 --version)${NC}"

command -v ffmpeg >/dev/null 2>&1 || { echo -e "${RED}❌ FFmpeg required. Install: sudo apt install ffmpeg${NC}"; exit 1; }
echo -e "${GREEN}✅ FFmpeg found: $(ffmpeg -version 2>&1 | head -1)${NC}"

# ─── Setup .env ───────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env from template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env and add your ANTHROPIC_API_KEY${NC}"
    echo -e "${YELLOW}   Get one free at: https://console.anthropic.com/${NC}"
fi

# ─── Create virtual environment ───────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# ─── Install dependencies ─────────────────────────────────────────────────────
echo -e "${YELLOW}Installing dependencies (this may take a few minutes)...${NC}"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "${GREEN}✅ Dependencies installed${NC}"

# ─── Create storage dirs ──────────────────────────────────────────────────────
mkdir -p storage/uploads storage/outputs storage/temp
echo -e "${GREEN}✅ Storage directories created${NC}"

# ─── Start services ───────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}🚀 Starting AttentionX...${NC}"
echo ""

# Start backend in background
echo -e "${BLUE}Starting Backend API on http://localhost:8000${NC}"
cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to start...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is running!${NC}"
        break
    fi
    sleep 1
done

# Start frontend
echo -e "${BLUE}Starting Frontend on http://localhost:8501${NC}"
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  🎬 AttentionX is ready!                ║${NC}"
echo -e "${GREEN}║  Frontend: http://localhost:8501        ║${NC}"
echo -e "${GREEN}║  API Docs:  http://localhost:8000/docs  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"

# Start Streamlit (foreground)
cd frontend && streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --browser.gatherUsageStats=false \
    --theme.base=dark \
    --theme.primaryColor="#7c3aed"

# Cleanup on exit
kill $BACKEND_PID 2>/dev/null
