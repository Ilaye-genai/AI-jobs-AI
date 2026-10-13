# The VOICE AGENT (voice_agent.py). It is a long-running worker, so it cannot run on Vercel.
# Use this on LiveKit Cloud (`lk agent create`), Railway, Fly.io, Render (background worker) or any Docker host.
# Secrets (DEEPSEEK_API_KEY, DEEPGRAM_API_KEY, LIVEKIT_*, QDRANT_*, ...) are injected at runtime by the host:
# never copy a .env file into the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# CA certificates: the LiveKit SDK needs them to reach LiveKit Cloud
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos "" --uid 10001 appuser
WORKDIR /app

COPY requirements-agent.txt .
RUN pip install -r requirements-agent.txt

COPY --chown=appuser:appuser . .
USER appuser

# Download the search models now (about 300 MB) so the first visitor does not wait for them.
# Failing here is not fatal: voice_agent.py warms them up again at start-up.
RUN python -c "import vector_store as vs; vs.get_embedder().dim; r = vs.get_reranker(); r and r.model" || true

# A fixed command that starts the worker: it connects out to LiveKit and waits for visitors to join.
CMD ["python", "voice_agent.py", "start"]
