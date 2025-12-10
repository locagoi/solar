@echo off
echo Removing old Solar app image if it exists...
docker rmi solar-app 2>nul

echo Building Solar app Docker image...
docker build -t solar-app .

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)

echo Stopping and removing existing container if it exists...
docker stop solar-dev 2>nul
docker rm solar-dev 2>nul

echo Starting Solar app in development mode...
docker run -d ^
  -p 8000:8000 ^
  -v "%cd%\app:/app/app" ^
  -v "%cd%\requirements.txt:/app/requirements.txt" ^
  -v "%cd%\.env:/app/.env" ^
  -e LOG_LEVEL=DEBUG ^
  -e PYTHONPATH=/app ^
  --env-file .env ^
  --name solar-dev ^
  solar-app ^
  /venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --access-log

if %ERRORLEVEL% NEQ 0 (
    echo Failed to start container!
    exit /b %ERRORLEVEL%
)

echo Container started. Showing logs (Press Ctrl+C to stop)...
docker logs -f solar-dev
