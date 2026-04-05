# Saido Agent Deployment Guide

## Local Development with Docker Compose

### Prerequisites
- Docker Engine 20.10+
- Docker Compose v2+

### Quick Start

1. Copy the environment file and configure it:
   ```bash
   cp .env.example .env
   # Edit .env with your production values
   ```

2. Build and start the service:
   ```bash
   docker compose up --build
   ```

3. Verify the service is running:
   ```bash
   curl http://localhost:8000/health
   # Expected: {"status":"healthy","version":"0.1.0"}
   ```

4. Stop the service:
   ```bash
   docker compose down
   ```

### Persistent Data

Docker Compose defines two named volumes:
- `saido-data` — mounted at `/data` (knowledge directory)
- `saido-db` — mounted at `/home/saido/.saido_agent` (SQLite database, API keys)

To reset all data:
```bash
docker compose down -v
```

---

## Production Deployment

### Building the Image

```bash
docker build -t saido-agent:latest .
```

The Dockerfile uses a multi-stage build:
- **Builder stage**: installs system dependencies (gcc, libffi-dev), builds the Python wheel
- **Runtime stage**: copies only the virtual environment and application source, runs as non-root user `saido` (UID 1000)

### Running in Production

```bash
docker run -d \
  --name saido-agent \
  -p 8000:8000 \
  -e SAIDO_JWT_SECRET="$(openssl rand -hex 32)" \
  -e SAIDO_KNOWLEDGE_DIR=/data/knowledge \
  -e SAIDO_API_PORT=8000 \
  -e SAIDO_LOG_LEVEL=warning \
  -v saido-data:/data \
  -v saido-db:/home/saido/.saido_agent \
  --restart unless-stopped \
  saido-agent:latest
```

### Database Migrations

Migrations are stored in `saido_agent/api/migrations/` as numbered SQL files. Apply them programmatically:

```python
from saido_agent.api.db import run_migrations
run_migrations()
```

Or via the CLI before starting the server. Migrations are idempotent and tracked in the `_migrations` table.

---

## Environment Variables Reference

| Variable | Description | Default | Required |
|---|---|---|---|
| `SAIDO_JWT_SECRET` | Secret key for JWT token signing | (auto-generated) | Yes (production) |
| `SAIDO_KNOWLEDGE_DIR` | Path to knowledge file storage | `/data/knowledge` | No |
| `SAIDO_API_PORT` | Port the API server listens on | `8000` | No |
| `SAIDO_LOG_LEVEL` | Logging level (debug/info/warning/error) | `info` | No |

**Security notes:**
- Always set `SAIDO_JWT_SECRET` to a strong random value in production
- Never use the default `dev-secret-change-me` outside local development
- The `.env` file should never be committed to version control

---

## Health Check Endpoint

**Endpoint:** `GET /health`

**Response:**
```json
{"status": "healthy", "version": "0.1.0"}
```

**Usage:**
- Docker HEALTHCHECK is configured in the Dockerfile (30s interval, 5s timeout, 3 retries)
- Load balancers should poll this endpoint to determine instance health
- Returns HTTP 200 when the service is ready to accept requests

---

## Architecture Notes

- **Non-root execution**: The container runs as user `saido` (UID 1000) for security
- **SQLite database**: Located at `~/.saido_agent/saido.db` inside the container. For multi-node deployments, replace with PostgreSQL
- **Stateless API**: JWT tokens enable horizontal scaling. Agent state is cached in-memory per instance
- **CORS**: Defaults to allow all origins. Restrict via environment or middleware configuration for production
