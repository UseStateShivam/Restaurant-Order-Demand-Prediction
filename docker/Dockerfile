# Production-grade Dockerfile for Restaurant Order Demand Prediction
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications first to leverage Docker layer caching
COPY pyproject.toml requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code and configurations
COPY configs/ ./configs/
COPY src/ ./src/
COPY models/ ./models/
COPY data/raw/*.csv ./data/raw/
COPY app.py ./

# Install package in editable/local mode
RUN pip install -e .

# Create non-root system user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

# Health check to ensure service vitality
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command starts FastAPI prediction service
CMD ["uvicorn", "restaurant_demand.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
