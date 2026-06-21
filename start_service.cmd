@echo off
echo Starting RAG MCP Backend...
cd /d "%~dp0backend"
..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
