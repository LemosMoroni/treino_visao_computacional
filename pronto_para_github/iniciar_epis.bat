@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp007_testar_epis.py" %*
if errorlevel 1 pause
