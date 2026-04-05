FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for cryptography wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY saido_agent/ saido_agent/

RUN pip install --no-cache-dir .

EXPOSE 8000

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "saido_agent.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
