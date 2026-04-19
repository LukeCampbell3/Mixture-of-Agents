FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY configs/ ./configs/
COPY setup.py .
COPY README.md .

# Create data directory
RUN mkdir -p /app/data

# Install package
RUN pip install -e .

# Expose port for API (if we add one later)
EXPOSE 5000

# Default command
CMD ["python", "-m", "app.main", "--help"]
