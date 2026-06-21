@echo off
echo Starting Chainlit UI...
cd /d "%~dp0chainlit_app"
..\.venv\Scripts\python.exe -m chainlit run app.py --host 0.0.0.0 --port 8001
