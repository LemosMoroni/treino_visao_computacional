@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0dashboard_epis\server.py" --abrir %*
if errorlevel 1 pause
