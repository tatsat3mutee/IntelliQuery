# IntelliQuery Simple - Run Script
# Sets up environment and starts the server

Write-Host "🎯 IntelliQuery Simple - Starting..." -ForegroundColor Green

# Set PYTHONPATH
$env:PYTHONPATH = "src"
Write-Host "📂 PYTHONPATH set to: src" -ForegroundColor Cyan

# Check if .env exists
if (!(Test-Path ".env")) {
    Write-Host "⚠️ WARNING: .env file not found!" -ForegroundColor Yellow
    Write-Host "   Create .env with your Databricks credentials" -ForegroundColor Yellow
}

# Start server
Write-Host "🚀 Starting FastAPI server on http://localhost:8001" -ForegroundColor Green
uvicorn IntelliQuery_simple.app:app --host 0.0.0.0 --port 8001 --reload
