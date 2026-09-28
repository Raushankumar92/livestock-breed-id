@echo off
title Livestock Breed ID - Launcher

echo Starting Livestock Breed ID...

start "Livestock Breed ID - Backend" cmd /k "cd /d D:\livestock-breed-id\backend && D:\livestock-breed-id\.venv\Scripts\activate.bat && uvicorn app.main:app --reload --port 8000"

timeout /t 5 /nobreak >nul

start "Livestock Breed ID - Frontend" cmd /k "cd /d D:\livestock-breed-id\frontend && npm run dev"

timeout /t 5 /nobreak >nul

start http://localhost:5173/

echo.
echo Livestock Breed ID is starting...
echo.
pause