# PowerShell script to setup Ollama with Qwen2.5

param(
    [string]$ModelSize = "7b"
)

Write-Host "Setting up Ollama with Qwen2.5..." -ForegroundColor Green

# Check if Ollama is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -ErrorAction Stop
    Write-Host "Ollama is running." -ForegroundColor Green
} catch {
    Write-Host "Error: Ollama is not running. Please start Ollama first." -ForegroundColor Red
    Write-Host "Run: docker-compose up -d ollama" -ForegroundColor Yellow
    exit 1
}

$ModelName = "qwen2.5:$ModelSize"
Write-Host "Pulling $ModelName..." -ForegroundColor Green

# Pull Qwen2.5 model
docker exec agentic-ollama ollama pull $ModelName

if ($LASTEXITCODE -eq 0) {
    Write-Host "Model pulled successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Testing model..." -ForegroundColor Green
    docker exec agentic-ollama ollama run $ModelName "Hello, how are you?"
    
    Write-Host ""
    Write-Host "Setup complete! You can now use Qwen2.5 with the agentic network." -ForegroundColor Green
    Write-Host "Model: $ModelName" -ForegroundColor Cyan
    Write-Host "API endpoint: http://localhost:11434" -ForegroundColor Cyan
} else {
    Write-Host "Error pulling model." -ForegroundColor Red
    exit 1
}
