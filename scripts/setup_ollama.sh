#!/bin/bash
# Setup script for Ollama with Qwen2.5

set -e

echo "Setting up Ollama with Qwen2.5..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Error: Ollama is not running. Please start Ollama first."
    echo "Run: docker-compose up -d ollama"
    exit 1
fi

echo "Ollama is running. Pulling Qwen2.5 model..."

# Pull Qwen2.5 model (adjust size as needed)
# Options: qwen2.5:0.5b, qwen2.5:1.5b, qwen2.5:3b, qwen2.5:7b, qwen2.5:14b, qwen2.5:32b, qwen2.5:72b
MODEL_SIZE=${1:-7b}
MODEL_NAME="qwen2.5:${MODEL_SIZE}"

echo "Pulling ${MODEL_NAME}..."
docker exec agentic-ollama ollama pull ${MODEL_NAME}

echo "Model pulled successfully!"
echo ""
echo "Testing model..."
docker exec agentic-ollama ollama run ${MODEL_NAME} "Hello, how are you?"

echo ""
echo "Setup complete! You can now use Qwen2.5 with the agentic network."
echo "Model: ${MODEL_NAME}"
echo "API endpoint: http://localhost:11434"
