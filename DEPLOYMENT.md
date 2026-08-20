# Deployment Guide — Enterprise RAG Assistant

A complete guide to deploying the Enterprise RAG Assistant to production, from platform selection to monitoring.

---

## Table of Contents

1. [Platform Selection](#1-platform-selection)
2. [Prerequisites](#2-prerequisites)
3. [Pre-Deployment Checklist](#3-pre-deployment-checklist)
4. [Option A: Railway (Recommended)](#4-option-a-railway-recommended)
5. [Option B: Render](#5-option-b-render)
6. [Option C: VPS (DigitalOcean / AWS EC2)](#6-option-c-vps-digitalocean--aws-ec2)
7. [Environment Variable Configuration](#7-environment-variable-configuration)
8. [Domain & SSL Setup](#8-domain--ssl-setup)
9. [CI/CD Pipeline](#9-cicd-pipeline)
10. [Monitoring & Logging](#10-monitoring--logging)
11. [Scaling Considerations](#11-scaling-considerations)
12. [Rollback Strategy](#12-rollback-strategy)
13. [Production Best Practices](#13-production-best-practices)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Platform Selection

### Architecture Constraints

Before choosing a platform, understand what this application requires:

| Requirement | Reason | Impact |
|-------------|--------|--------|
| **Persistent disk storage** | ChromaDB (vectors) and, on a single node, the SQLite stores and NetworkX JSON knowledge graph are file-based | Eliminates serverless platforms without persistent storage |
| **Long-running requests** | LLM calls take 5-30s; RAGAS evaluation takes minutes | Eliminates platforms with <30s request timeouts |
| **2+ GB RAM** | `sentence-transformers` loads PyTorch + cross-encoder model (~200MB model weights) | Eliminates free tiers with <512MB RAM |
| **Two services** | FastAPI API (port 8000) + Streamlit UI (port 8501) | Platform must support multi-service or multiple apps |
| **SSE streaming** | Real-time token streaming via Server-Sent Events | Needs HTTP/1.1 chunked transfer support |
| **File uploads** | Users upload PDF/TXT/MD documents via `/upload` | Needs writable filesystem |
| **Large Docker image** | PyTorch + sentence-transformers + all deps = ~3-4 GB image | Needs sufficient build resources and registry storage |

### Platform Comparison

| Platform | Persistent Storage | Long Requests | Multi-Service | RAM | Ease of Setup | Monthly Cost |
|----------|--------------------|---------------|---------------|-----|--------------|-------------|
| **Railway** | Volumes | No limit | Monorepo support | Up to 32 GB | Very Easy | ~$10-25 |
| **Render** | Persistent Disks | No limit (paid) | Multi-service | Up to 64 GB | Easy | ~$15-30 |
| **DigitalOcean Droplet** | Full disk | No limit | Docker Compose | Up to 256 GB | Moderate | ~$12-24 |
| **AWS EC2** | EBS volumes | No limit | Docker Compose | Any | Complex | ~$15-40 |
| **Fly.io** | Volumes | No limit | Multi-app | Up to 64 GB | Moderate | ~$10-20 |
| **Vercel** | None | 10-60s limit | No backend | 1 GB | N/A | N/A |
| **GCP Cloud Run** | None (ephemeral) | 60min limit | Per-service | 32 GB | Moderate | ~$10-30 |

### Verdict

| Rank | Platform | Why |
|------|----------|-----|
| 1 | **Railway** | Easiest setup, handles all constraints, GitHub auto-deploy, persistent volumes, no timeout limits. Best for beginners. |
| 2 | **Render** | Similar to Railway, slightly more manual setup for multi-service. Good free tier for testing (has spin-down). |
| 3 | **VPS (DigitalOcean/EC2)** | Full control, run `docker-compose` directly. Best if you want to learn infrastructure. |

> **Recommendation**: Use **Railway** for the fastest path to production. Use a **VPS** if you want full control and lower long-term costs.

> **Avoid**: Vercel (serverless, no persistent storage, no long-running processes), GCP Cloud Run (no persistent disk), Heroku (ephemeral filesystem, SQLite data lost on restart).

### Before running more than one replica

Everything above assumes a single node. Four pieces of state are
per-process by default, and each fails *silently* rather than loudly when
a second replica appears — which is why they are listed here rather than
left to be discovered:

| Set this | Otherwise |
|----------|-----------|
| `DATABASE_URL=postgresql://…` | Each replica keeps its own document registry (the same document indexes twice) and its own conversation memory (a follow-up question routed elsewhere forgets the thread). Metrics report only the traffic that replica served. |
| `RATE_LIMIT_STORAGE_URI=redis://…` | Counters are per-process, so N replicas allow N x the configured limit, and a restart resets them. |
| `CHROMA_MODE=server` | Embedded ChromaDB cannot see another process's writes; the API never serves a document the worker indexed. |
| `EVENT_BUS=kafka` | The SQLite queue is durable but local — workers must share one `EVENT_BUS_PATH`, which means one host. |

`docker-compose.prod.yml` sets all four. The API logs a warning at
startup when it detects the single-node configuration together with auth
or async ingestion enabled, so a deployment that believes it is scaled
can see that it is not.

---

## 2. Prerequisites

### Accounts Required

- [ ] **GitHub account** — your code repository (Railway/Render deploy from GitHub)
- [ ] **OpenAI account** — API key for LLM and embeddings ([platform.openai.com](https://platform.openai.com))
- [ ] **Deployment platform account** — Railway, Render, or cloud provider
- [ ] **Domain name** (optional) — for custom domain (e.g., from Namecheap, Cloudflare, GoDaddy)

### API Keys Ready

- [ ] `OPENAI_API_KEY` — **Required**. Get from OpenAI dashboard > API Keys
- [ ] `TAVILY_API_KEY` — Optional. For web search fallback ([tavily.com](https://tavily.com))
- [ ] `LANGSMITH_API_KEY` — Optional. For distributed tracing ([smith.langchain.com](https://smith.langchain.com))
- [ ] `API_KEYS` — Generate your own. Any random string works (e.g., use `openssl rand -hex 32`)

### Tools Installed (for VPS option)

```bash
# Check versions
git --version          # >= 2.30
docker --version       # >= 24.0
docker compose version # >= 2.20
```

---

## 3. Pre-Deployment Checklist

Before deploying, verify your application works locally:

```bash
# 1. All tests pass
pytest

# 2. API starts and serves health check
uvicorn api.app:app --port 8000 &
curl http://localhost:8000/health
# Should return: {"status": "healthy", ...}

# 3. Documents are ingested
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "./data/sample_docs"}'

# 4. Query works end-to-end
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the leave policy?", "mode": "graph"}'
```

### Push Code to GitHub

```bash
# Initialize repo if not already done
git init
git add -A
git commit -m "Production-ready deployment"
git remote add origin https://github.com/YOUR_USERNAME/enterprise-rag-assistant.git
git push -u origin main
```

> **Important**: Verify `.env` is in `.gitignore`. Never push secrets to GitHub.

---

## 4. Option A: Railway (Recommended)

Railway is the fastest path from local to production. It supports Docker, persistent volumes, environment variables, and automatic deploys from GitHub.

### Step 1: Create Railway Account

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub (this links your repositories)
3. Add a payment method (Hobby plan: $5/month + usage)

### Step 2: Create New Project

1. Click **"New Project"** on dashboard
2. Select **"Deploy from GitHub Repo"**
3. Choose `enterprise-rag-assistant` from the list
4. Railway will auto-detect the `Dockerfile` and start building

### Step 3: Configure the API Service

1. Click on the service that was created
2. Go to **Settings** tab:
   - **Service Name**: `rag-api`
   - **Build**: Docker
   - **Dockerfile Path**: `Dockerfile.prod`
   - **Watch Paths**: `/api, /src, /config.py, /requirements.txt, /Dockerfile.prod`
3. Go to **Networking** tab:
   - Click **"Generate Domain"** — Railway gives you a public URL like `rag-api-production-xxxx.up.railway.app`
   - Note this URL — you'll need it for the UI service

### Step 4: Add Persistent Volume

1. In the **rag-api** service, go to **Volumes** tab
2. Click **"Add Volume"**
3. Create two volumes:

| Volume Name | Mount Path | Size |
|-------------|------------|------|
| `chroma-data` | `/app/chroma_db` | 1 GB |
| `checkpoint-data` | `/app/checkpoints` | 500 MB |

### Step 5: Set Environment Variables

1. Go to the **Variables** tab
2. Click **"Raw Editor"** and paste:

```
OPENAI_API_KEY=sk-your-actual-key-here
AUTH_ENABLED=true
API_KEYS=your-generated-api-key-here
GUARDRAILS_ENABLED=true
PII_DETECTION_ENABLED=true
DEBUG_MODE=false
LOG_LEVEL=WARNING
CORS_ORIGINS=https://rag-ui-production-xxxx.up.railway.app
CHROMA_DIR=/app/chroma_db
CHECKPOINT_DIR=/app/checkpoints
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
MEMORY_ENABLED=true
```

> Replace the placeholder values with your actual keys. See [Section 7](#7-environment-variable-configuration) for the full list.

### Step 6: Deploy the UI Service

1. In the same project, click **"New"** > **"Service"** > **"GitHub Repo"**
2. Select the same repository
3. Go to **Settings**:
   - **Service Name**: `rag-ui`
   - **Dockerfile Path**: `Dockerfile.prod`
   - **Custom Start Command**: `streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true`
4. Go to **Networking** > **"Generate Domain"**
5. Set **Variables**:

```
API_URL=https://rag-api-production-xxxx.up.railway.app
OPENAI_API_KEY=sk-your-actual-key-here
```

> Replace `rag-api-production-xxxx.up.railway.app` with the actual API domain from Step 3.

### Step 7: Ingest Documents

After both services are running:

```bash
# Ingest sample documents via API
curl -X POST https://rag-api-production-xxxx.up.railway.app/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-generated-api-key-here" \
  -d '{"path": "./data/sample_docs"}'
```

Or upload files via the Streamlit UI at your UI domain.

### Step 8: Verify Deployment

```bash
# Health check
curl https://rag-api-production-xxxx.up.railway.app/health

# Ask a question
curl -X POST https://rag-api-production-xxxx.up.railway.app/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-generated-api-key-here" \
  -d '{"question": "What is the remote work policy?", "mode": "graph"}'
```

Visit the Streamlit UI in your browser to verify the full experience.

---

## 5. Option B: Render

### Step 1: Create Render Account

1. Go to [render.com](https://render.com)
2. Sign up with GitHub

### Step 2: Create API Service

1. Click **"New"** > **"Web Service"**
2. Connect your GitHub repo
3. Configure:
   - **Name**: `rag-api`
   - **Runtime**: Docker
   - **Dockerfile Path**: `Dockerfile.prod`
   - **Instance Type**: Standard ($7/month) or higher
   - **Region**: Choose closest to your users

### Step 3: Add Persistent Disk

1. In the service settings, scroll to **"Disks"**
2. Add a disk:
   - **Name**: `rag-storage`
   - **Mount Path**: `/app/storage`
   - **Size**: 1 GB

3. Update environment variables:
   ```
   CHROMA_DIR=/app/storage/chroma_db
   CHECKPOINT_DIR=/app/storage/checkpoints
   ```

### Step 4: Set Environment Variables

In the **"Environment"** section, add all variables from [Section 7](#7-environment-variable-configuration).

### Step 5: Create UI Service

1. Click **"New"** > **"Web Service"** again
2. Same repo, but configure:
   - **Name**: `rag-ui`
   - **Dockerfile Path**: `Dockerfile.prod`
   - **Docker Command**: `streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true`
3. Set environment variable:
   ```
   API_URL=https://rag-api.onrender.com
   ```

### Step 6: Verify

Same verification steps as Railway (Step 8 above), using your Render URLs.

> **Note**: Render's free tier spins down after 15 minutes of inactivity. Use a paid plan ($7/month) for always-on services.

---

## 6. Option C: VPS (DigitalOcean / AWS EC2)

For full control over infrastructure. Recommended for teams with DevOps experience or for learning infrastructure management.

### Step 1: Provision a Server

**DigitalOcean Droplet**:
1. Go to [digitalocean.com](https://www.digitalocean.com)
2. Create a Droplet:
   - **Image**: Ubuntu 24.04
   - **Plan**: Basic $12/month (2 vCPU, 2 GB RAM) minimum; $24/month (4 GB RAM) recommended
   - **Region**: Closest to your users
   - **Authentication**: SSH key (generate one with `ssh-keygen -t ed25519`)

**AWS EC2**:
1. Go to AWS Console > EC2 > Launch Instance
2. Configure:
   - **AMI**: Ubuntu 24.04
   - **Instance type**: `t3.medium` (2 vCPU, 4 GB RAM) — ~$30/month
   - **Storage**: 20 GB gp3
   - **Security group**: Open ports 22 (SSH), 80 (HTTP), 443 (HTTPS)

### Step 2: SSH Into Server

```bash
ssh root@YOUR_SERVER_IP
# or for EC2:
ssh -i your-key.pem ubuntu@YOUR_SERVER_IP
```

### Step 3: Install Docker

```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose plugin
apt install -y docker-compose-plugin

# Verify installation
docker --version
docker compose version

# (Optional) Add your user to docker group to avoid sudo
usermod -aG docker $USER
```

### Step 4: Clone Repository

```bash
cd /opt
git clone https://github.com/YOUR_USERNAME/enterprise-rag-assistant.git
cd enterprise-rag-assistant
```

### Step 5: Configure Environment

```bash
cp .env.example .env
nano .env
```

Fill in all required variables (see [Section 7](#7-environment-variable-configuration)). At minimum:

```bash
OPENAI_API_KEY=sk-your-actual-key-here
AUTH_ENABLED=true
API_KEYS=your-generated-api-key-here
DEBUG_MODE=false
LOG_LEVEL=WARNING
CORS_ORIGINS=https://your-domain.com
```

### Step 6: Deploy with Docker Compose

```bash
# Build and start in detached mode
docker compose -f docker-compose.prod.yml up --build -d

# Check service status
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f

# Verify health
curl http://localhost:8000/health
```

### Step 7: Set Up Nginx Reverse Proxy (for domain + SSL)

```bash
# Install Nginx and Certbot
apt install -y nginx certbot python3-certbot-nginx
```

Create Nginx config:

```bash
cat > /etc/nginx/sites-available/rag-assistant << 'NGINX'
# API backend
server {
    listen 80;
    server_name api.your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE streaming support
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    # File upload size
    client_max_body_size 15M;
}

# Streamlit UI
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (required for Streamlit)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
NGINX
```

Enable the config and get SSL:

```bash
# Enable site
ln -s /etc/nginx/sites-available/rag-assistant /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default

# Test config
nginx -t

# Restart Nginx
systemctl restart nginx

# Get SSL certificate (replace with your domains)
certbot --nginx -d your-domain.com -d api.your-domain.com

# Auto-renew SSL
certbot renew --dry-run
```

### Step 8: Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"path": "./data/sample_docs"}'
```

### Step 9: Set Up Auto-Restart

```bash
# Docker Compose services already have restart: unless-stopped
# But also enable Docker to start on boot:
systemctl enable docker
```

### Step 10: Set Up Firewall

```bash
# Allow only necessary ports
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw enable

# Block direct access to application ports (only Nginx should access them)
# Ports 8000 and 8501 are NOT opened — Nginx proxies to them internally
```

---

## 7. Environment Variable Configuration

### Required Variables

| Variable | Value | Notes |
|----------|-------|-------|
| `OPENAI_API_KEY` | `sk-...` | From OpenAI dashboard. **Never commit this.** |

### Production Security (Strongly Recommended)

| Variable | Value | Notes |
|----------|-------|-------|
| `AUTH_ENABLED` | `true` | Protects all endpoints except `/health` |
| `API_KEYS` | `key1,key2,...` | Generate with `openssl rand -hex 32`. Comma-separated for multiple keys |
| `GUARDRAILS_ENABLED` | `true` | Blocks prompt injection and PII in queries |
| `PII_DETECTION_ENABLED` | `true` | Redacts PII in LLM responses |
| `DEBUG_MODE` | `false` | Hides internal error details from API responses |
| `CORS_ORIGINS` | `https://your-domain.com` | Lock to your UI domain. Comma-separated for multiple |
| `MAX_QUERY_LENGTH` | `2000` | Prevents oversized query abuse |

### Performance Tuning

| Variable | Value | Notes |
|----------|-------|-------|
| `LOG_LEVEL` | `WARNING` | Reduce log volume in production |
| `LLM_TIMEOUT` | `30` | Seconds before LLM call times out |
| `LLM_MAX_RETRIES` | `2` | Retries on transient LLM failures |
| `RATE_LIMIT_PER_MINUTE` | `30/minute` | Adjust based on expected traffic |
| `COST_BUDGET_PER_QUERY` | `0.02` | Maximum cost per query (USD) |

### Feature Flags

| Variable | Default | Production Recommendation |
|----------|---------|--------------------------|
| `MEMORY_ENABLED` | `true` | Keep `true` for multi-turn conversations |
| `SEMANTIC_CACHE_ENABLED` | `false` | Set `true` to reduce latency and API costs |
| `KNOWLEDGE_GRAPH_ENABLED` | `false` | Set `true` if you need multi-hop reasoning |
| `PARALLEL_SUB_QUERIES` | `false` | Set `true` for faster complex question processing |
| `INTENT_DETECTION_ENABLED` | `true` | Keep `true` for smart query routing |

### Storage Paths

| Variable | Docker Value | Notes |
|----------|-------------|-------|
| `CHROMA_DIR` | `/app/chroma_db` | Must point to persistent volume |
| `CHECKPOINT_DIR` | `/app/checkpoints` | Must point to persistent volume |
| `KG_PERSIST_PATH` | `/app/checkpoints/knowledge_graph.json` | Inside checkpoint volume |

### Optional Integrations

| Variable | Notes |
|----------|-------|
| `TAVILY_API_KEY` | Enables web search fallback when documents are insufficient |
| `LANGSMITH_TRACING` | Set `true` + provide `LANGSMITH_API_KEY` for distributed tracing |
| `LANGSMITH_PROJECT` | Project name in LangSmith dashboard |

---

## 8. Domain & SSL Setup

### Option A: Platform-Provided Domains (Railway / Render)

Both Railway and Render provide free subdomains with automatic SSL:
- Railway: `your-service.up.railway.app`
- Render: `your-service.onrender.com`

No additional setup needed.

### Option B: Custom Domain

#### 1. Purchase a Domain

Use any registrar (Namecheap, Cloudflare, GoDaddy, etc.). Example: `rag-assistant.com`

#### 2. Configure DNS Records

Add these DNS records at your registrar:

| Type | Name | Value | TTL |
|------|------|-------|-----|
| A / CNAME | `@` (root) | Platform-provided URL or server IP | 300 |
| A / CNAME | `api` | Platform-provided URL or server IP | 300 |

**For Railway/Render**: Use CNAME pointing to the platform domain.
**For VPS**: Use A record pointing to your server's IP address.

#### 3. Add Domain in Platform

**Railway**: Service > Settings > Networking > Custom Domain > add your domain
**Render**: Service > Settings > Custom Domains > add your domain

#### 4. SSL Certificate

- **Railway/Render**: Automatic SSL via Let's Encrypt. No action needed.
- **VPS**: Use Certbot (covered in [VPS Step 7](#step-7-set-up-nginx-reverse-proxy-for-domain--ssl))

#### 5. Update CORS

After setting up the domain, update `CORS_ORIGINS` to include your domain:

```
CORS_ORIGINS=https://your-domain.com,https://api.your-domain.com
```

---

## 9. CI/CD Pipeline

### GitHub Actions (Works with Any Platform)

Create `.github/workflows/deploy.yml`:

```yaml
name: Test & Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest --tb=short -q
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # --- Railway deployment ---
      - name: Deploy to Railway
        uses: bervProject/railway-deploy@main
        with:
          railway_token: ${{ secrets.RAILWAY_TOKEN }}
          service: rag-api

      # --- OR: Render deployment (uncomment below, comment Railway above) ---
      # - name: Deploy to Render
      #   run: |
      #     curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_URL }}"
```

### Setup

1. Go to your GitHub repo > **Settings** > **Secrets and variables** > **Actions**
2. Add secrets:
   - `OPENAI_API_KEY` — for tests
   - `RAILWAY_TOKEN` — from Railway dashboard > Account > Tokens
   - (or `RENDER_DEPLOY_HOOK_URL` — from Render service > Settings > Deploy Hook)

### Railway Auto-Deploy

Railway auto-deploys on every push to `main` by default. The GitHub Action above adds a test gate — deploys only happen after tests pass.

To disable Railway's auto-deploy (so only GitHub Actions triggers it):
1. Railway service > Settings > **Deploy** > Disable "Auto Deploy"

---

## 10. Monitoring & Logging

### Application-Level Monitoring

The application has built-in observability:

```bash
# Health check (basic)
curl https://api.your-domain.com/health

# Deep health check (verifies ChromaDB, SQLite, LLM connectivity)
curl "https://api.your-domain.com/health?deep=true"

# Query metrics (via CLI on server)
docker compose -f docker-compose.prod.yml exec api python -m scripts.metrics --last 50
```

### Prometheus & alerting

`GET /metrics` exposes cost, latency percentiles, IDK rate, queue depth,
dead-letter depth and per-status document counts in Prometheus exposition
format. It requires an API key for the same reason `?deep=true` does —
the body carries queue depths and per-department document counts.

```bash
curl -H "Authorization: Bearer $RAG_API_KEY" https://api.your-domain.com/metrics
```

A scrape config and alert rules ship in `deploy/prometheus/`:

```bash
docker run -p 9090:9090 \
  -v "$PWD/deploy/prometheus:/etc/prometheus" \
  prom/prometheus:v2.55.1
```

The rules cover the conditions that otherwise need a person to go
looking. Two are worth calling out because they are specific to a RAG
system rather than to web services generally:

- **`RagDeadLetterQueueNonEmpty`** — a dead-lettered document is not
  slow, it is never going to be indexed. Nobody notices until someone
  asks a question it should have answered.
- **`RagIdkRateSpike`** — a rising "I don't know" rate almost never means
  the questions got harder. It means retrieval broke: an empty
  collection, a changed embedding model, a scoping change that filtered
  everything out. All of those look healthy to every other probe.

### Distributed tracing (optional)

`OTEL_ENABLED=true` with `OTEL_EXPORTER_OTLP_ENDPOINT` emits spans, and
propagates the trace context *inside the ingestion event* — so an upload
and the indexing that happens minutes later in the worker process appear
as one trace rather than two unrelated ones. With it off, the SDK is
never imported.

### Log Management

```bash
# View live logs
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f ui

# View last 100 lines
docker compose -f docker-compose.prod.yml logs --tail 100 api
```

**Production log level**: Set `LOG_LEVEL=WARNING` to reduce noise. Set to `INFO` temporarily for debugging.

### External Monitoring (Recommended)

#### Uptime Monitoring

Use a free uptime monitor to alert you if the service goes down:

- **UptimeRobot** (free, 5-min checks): [uptimerobot.com](https://uptimerobot.com)
  - Add HTTP monitor: `GET https://api.your-domain.com/health`
  - Expected status: 200
  - Check interval: 5 minutes
  - Alert: Email/Slack on failure

- **Better Stack** (free tier): [betterstack.com](https://betterstack.com)

#### LangSmith Tracing (Recommended)

Enable distributed tracing for LLM call debugging:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls-your-key-here
LANGSMITH_PROJECT=enterprise-rag-assistant
```

This gives you:
- Every LLM call with input/output
- Token counts and costs
- Latency breakdown per node
- Error traces

Dashboard: [smith.langchain.com](https://smith.langchain.com)

### Cost Monitoring

The application tracks OpenAI costs per query in SQLite. Monitor spending:

```bash
# On the server
docker compose -f docker-compose.prod.yml exec api python -m scripts.metrics --all
```

Set `COST_BUDGET_PER_QUERY=0.02` to cap per-query spend.

Also set up OpenAI usage alerts:
1. Go to [platform.openai.com/account/billing](https://platform.openai.com/account/billing)
2. Set a monthly budget limit
3. Enable email alerts

---

## 11. Scaling Considerations

### Current Architecture Limits

| Component | Bottleneck | Max Capacity |
|-----------|-----------|-------------|
| SQLite (`DATABASE_URL` unset) | Single writer at a time | ~50 concurrent users |
| ChromaDB (embedded) | Single process | ~100K documents |
| Uvicorn (1 worker) | Latency-bound LLM calls | ~10 concurrent queries |
| Cross-encoder | CPU inference | ~5 reranking requests/sec |
| BM25 sparse index | Whole corpus in memory, per filter key | `BM25_MAX_DOCUMENTS` (200K default), then dense-only |

Measure rather than guess — `loadtest/locustfile.py` reports p50, p95,
failure rate and IDK rate, and exits non-zero on the same thresholds the
Prometheus alerts use:

```bash
pip install locust
locust -f loadtest/locustfile.py --host http://localhost:8000 \
       --headless -u 20 -r 2 -t 5m --csv loadtest/results
```

### When to Scale

| Signal | Action |
|--------|--------|
| `rag_query_latency_ms{quantile="0.95"}` > 15s | Increase RAM, add API replicas (see the multi-replica settings above) |
| SQLite lock errors in logs | Set `DATABASE_URL` to PostgreSQL |
| ChromaDB query latency > 500ms | Switch to managed Pinecone/Weaviate |
| `rag_ingestion_queue_depth` rising and not draining | `--scale worker=N`, or raise `WORKER_CONCURRENCY` |
| Uploads returning 503 | Backpressure is shedding load; the pipeline needs more workers |
| Corpus past `BM25_MAX_DOCUMENTS` | Move sparse retrieval server-side (OpenSearch/Elasticsearch) |
| OpenAI rate limits | Set `LLM_FALLBACK_PROVIDER` for failover; add request queuing |

### Vertical Scaling (Easy)

Increase server resources:

| Platform | How |
|----------|-----|
| Railway | Change plan tier in service settings |
| Render | Upgrade instance type |
| VPS | Resize droplet / change EC2 instance type |

### Horizontal Scaling (Advanced)

If you need to scale beyond a single instance, the following changes are required:

1. **Replace SQLite with PostgreSQL** for metrics, memory, and cache stores
2. **Replace embedded ChromaDB** with managed vector DB (Pinecone, Weaviate, Qdrant)
3. **Add Redis** for semantic cache (replace SQLite cache)
4. **Load balancer** in front of multiple API instances
5. **Shared storage** (S3/GCS) for uploaded documents

This is a significant architectural change. For most use cases, vertical scaling (bigger server) is sufficient.

---

## 12. Rollback Strategy

### Railway / Render

Both platforms keep deployment history:

1. Go to service dashboard
2. Click **"Deployments"** tab
3. Find the last working deployment
4. Click **"Redeploy"** or **"Rollback"**

### VPS (Docker Compose)

```bash
# See recent deployments (git commits)
cd /opt/enterprise-rag-assistant
git log --oneline -10

# Rollback to a specific commit
git checkout <commit-hash>

# Rebuild and restart
docker compose -f docker-compose.prod.yml up --build -d

# If you need to go back to latest
git checkout main
docker compose -f docker-compose.prod.yml up --build -d
```

### Backup and Restore

State lives in three Docker volumes, not inside the application containers:

| Volume | Contents | Rebuildable? |
|--------|----------|--------------|
| `chroma_data` | Vector index | Only by re-embedding everything — the expensive one |
| `checkpoint_data` | Document registry, metrics, conversation memory, event queue | No |
| `minio_data` | Original uploaded bytes | No — and required to re-index anything |

Kafka is deliberately **not** backed up: it holds in-flight events, all of
which are reconstructable from the registry plus object storage.

```bash
./scripts/backup.sh backup                    # -> backups/<timestamp>/
./scripts/backup.sh verify backups/<ts>       # prove it is restorable
./scripts/backup.sh restore backups/<ts>      # verify, stop, replace, restart
```

`verify` runs automatically before any restore. It lists each archive and
fails if one is empty, and checks the manifest's recorded document count.

> **Why this replaced the previous procedure.** The old command ran
> `tar` on `/app/chroma_db` *inside the api container*. Since the move to
> `CHROMA_MODE=server`, api no longer mounts that path — vectors live in
> the `chroma` service's volume. The command still exited 0 and produced a
> plausible-looking archive containing nothing. Backups appeared healthy
> right up until a restore returned an empty index. `verify` exists
> specifically so that class of silent failure cannot recur.

**Run `verify` on a schedule, not just after backing up.** A backup that has
never been restored is a hypothesis.

### Emergency: Full Reset

If everything is broken and you need a clean start:

```bash
# WARNING: This deletes all data (vectors, memory, metrics)
docker compose -f docker-compose.prod.yml down -v   # -v removes volumes
docker compose -f docker-compose.prod.yml up --build -d

# Re-ingest documents
curl -X POST https://api.your-domain.com/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"path": "./data/sample_docs"}'
```

---

## 13. Production Best Practices

### Security

- [ ] `AUTH_ENABLED=true` — never run production without authentication
- [ ] Generate strong API keys: `openssl rand -hex 32`
- [ ] `DEBUG_MODE=false` — never expose stack traces to users
- [ ] `CORS_ORIGINS` locked to your UI domain only
- [ ] `.env` file has restricted permissions: `chmod 600 .env`
- [ ] Rotate API keys periodically
- [ ] Set OpenAI billing alerts and monthly spending limits
- [ ] Monitor for prompt injection attempts in logs

### Performance

- [ ] `LOG_LEVEL=WARNING` — reduce log I/O
- [ ] `SEMANTIC_CACHE_ENABLED=true` — avoid redundant LLM calls
- [ ] `RATE_LIMIT_PER_MINUTE=30/minute` — adjust to your load
- [ ] Consider `CROSS_ENCODER_DEVICE=cuda` if GPU available
- [ ] Set `PARALLEL_SUB_QUERIES=true` for faster multi-part responses

### Reliability

- [ ] Set up uptime monitoring (UptimeRobot/Better Stack)
- [ ] Enable LangSmith tracing for debugging
- [ ] Schedule regular data backups (weekly minimum)
- [ ] Test rollback procedure before you need it
- [ ] Docker `restart: unless-stopped` is set (it is in compose files)

### Cost Control

- [ ] Use `gpt-4o-mini` (not `gpt-4o`) — 10x cheaper, sufficient for most queries
- [ ] `COST_BUDGET_PER_QUERY=0.02` — hard limit per query
- [ ] `SEMANTIC_CACHE_ENABLED=true` — cache repeated queries
- [ ] Monitor monthly OpenAI usage at platform.openai.com
- [ ] Set `TOP_K=4` — more docs = more tokens = more cost

---

## 14. Troubleshooting

### Build Failures

**"Killed" during pip install (OOM)**
The `sentence-transformers` dependency pulls PyTorch (~2 GB). Your build environment needs at least 2 GB RAM.
- Railway: builds usually have sufficient RAM
- Render: ensure you're on a paid plan
- VPS: add swap space:
  ```bash
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
  ```

**Docker build too slow**
First build downloads ~3 GB of dependencies. Subsequent builds use Docker layer cache and are faster. If the build times out:
- Increase build timeout in platform settings
- Build locally and push to a container registry instead

### Runtime Errors

**"OPENAI_API_KEY is not set"**
- Verify the environment variable is set in your deployment platform
- Check for typos in the variable name
- Railway: Variables tab > verify the key is saved

**"Connection refused" on Streamlit**
- Verify the API service is healthy first: `curl https://api-url/health`
- Check `API_URL` in the UI service points to the correct API URL
- For Railway: use the internal URL (`http://rag-api.railway.internal:8000`) instead of the public URL

**SSE streaming not working**
- Nginx: ensure `proxy_buffering off` is set
- Cloudflare: disable "Rocket Loader" and "Auto Minify"
- Railway/Render: SSE works out of the box

**ChromaDB errors after redeployment**
- Ensure the volume is mounted correctly
- Check volume wasn't deleted during redeployment
- Verify `CHROMA_DIR` matches the volume mount path

**SQLite "database is locked"**
- This happens under high concurrency (>50 concurrent requests)
- Reduce uvicorn workers to 1: `--workers 1`
- For higher scale, migrate to PostgreSQL

**High latency (>10s per query)**
- Check LLM response times in LangSmith traces
- Verify the cross-encoder model is loaded (first query is slow)
- Consider enabling semantic cache to skip LLM for repeated queries
- Check server RAM — if swapping, upgrade to more memory

**502 Bad Gateway (Nginx)**
- API server crashed. Check logs: `docker compose logs api`
- Common cause: OOM kill. Check with `dmesg | grep -i oom`
- Solution: increase memory limits in `docker-compose.prod.yml`

### Platform-Specific Issues

**Railway: "Volume not found"**
- Volumes are region-specific. Ensure the volume and service are in the same region.

**Render: "Suspended due to inactivity"**
- Free tier services spin down after 15 minutes. Upgrade to paid ($7/month) for always-on.

**VPS: Server unreachable after reboot**
- Ensure Docker starts on boot: `systemctl enable docker`
- Verify UFW isn't blocking ports: `ufw status`

### Getting Help

- Open an issue on the project repository
- Check OpenAI status: [status.openai.com](https://status.openai.com)
- Railway docs: [docs.railway.app](https://docs.railway.app)
- Render docs: [docs.render.com](https://docs.render.com)
