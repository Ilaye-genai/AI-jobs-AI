"""
Website server: serves the job board, a small jobs API, and LiveKit room tokens.

    On Vercel   nothing to run: Vercel finds `app` below, serves everything in public/ itself (CDN),
                and runs the /api and /token routes as one function. Set LIVEKIT_URL, LIVEKIT_API_KEY and
                LIVEKIT_API_SECRET in the Vercel project settings (there is no .env there).
    Locally     python server.py   (address printed on start; change it with HOST / PORT)

The voice agent (voice_agent.py) is NOT part of this server and cannot run on Vercel: it is a long-running
worker. Host it separately (see Dockerfile); it connects out to LiveKit, so this site never needs its address.

Endpoints
    GET /                    the job board (public/index.html)
    GET /api/health          deployment check: how many jobs were found, which LiveKit variables are set
    GET /api/jobs            filtered list. Params: q, category, domain, level, employment, company,
                             skill, min_pay, sort (new|pay), limit, offset, ids (comma list, keeps order)
    GET /api/jobs/{job_id}   one job with its full description
    GET /api/version         changes when scraped_jobs.json changes (the page polls this)
    GET /token               a LiveKit token that also dispatches the voice agent into the room

The page never receives job data from the agent. The agent only sends Job IDs over LiveKit,
and the page loads those jobs from /api/jobs, so scraped_jobs.json stays the single source of truth.
"""
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import jobs_data

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

AGENT_NAME = "jobs-agent"  # must match agent_name in voice_agent.py
PUBLIC_DIR = Path(__file__).parent / "public"  # on Vercel this folder is served by the CDN, not by Python

app = FastAPI(title="AI jobs board")


@app.get("/api/jobs")
def list_jobs(
    q: str = "",
    category: str | None = None,
    domain: str | None = None,
    level: str | None = None,
    employment: str | None = None,
    company: str | None = None,
    skill: str | None = None,
    min_pay: float | None = None,
    sort: str = Query("new", pattern="^(new|pay)$"),
    limit: int = Query(60, ge=1, le=200),
    offset: int = Query(0, ge=0),
    ids: str | None = None,
) -> dict:
    version = jobs_data.version()
    if ids:
        picked = jobs_data.get_many([i for i in ids.split(",") if i.strip()][:50])
        return {"version": version, "total": len(picked), "jobs": picked}
    total, page = jobs_data.search(
        q=q, category=category, domain=domain, level=level, employment=employment,
        company=company, skill=skill, min_pay=min_pay, sort=sort, limit=limit, offset=offset,
    )
    return {"version": version, "total": total, "jobs": page, "facets": jobs_data.facets()}


@app.get("/api/jobs/{job_id}")
def one_job(job_id: str) -> dict:
    job = jobs_data.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No job with that id")
    return {k: v for k, v in job.items() if k != "search_text"}


@app.get("/api/health")
def health() -> dict:
    """Open this first on a new deployment: it shows whether the data file and the settings made it."""
    total, _ = jobs_data.search(
        q="", category=None, domain=None, level=None, employment=None, company=None,
        skill=None, min_pay=None, sort="new", limit=1, offset=0,
    )
    return {
        "ok": total > 0,
        "jobs": total,
        "agent_name": AGENT_NAME,
        "livekit_settings_present": {k: bool(os.getenv(k)) for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")},
    }


@app.get("/api/version")
def version() -> dict:
    return {"version": jobs_data.version()}


@app.get("/token")
def token() -> dict:
    """Join a fresh private room and ask LiveKit to dispatch the voice agent into it."""
    from livekit import api  # imported here so the board works even before livekit-api is installed

    url = os.getenv("LIVEKIT_URL")
    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")
    if not (url and key and secret):
        raise HTTPException(status_code=500, detail="LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in .env")

    room = f"jobs-{uuid.uuid4().hex[:10]}"
    jwt = (
        api.AccessToken(key, secret)
        .with_identity(f"visitor-{uuid.uuid4().hex[:8]}")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        # sync_streams keeps the avatar's lips in step with its voice
        .with_room_config(api.RoomConfiguration(sync_streams=True, agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)]))
        .to_jwt()
    )
    return {"url": url, "token": jwt, "room": room}


# Locally Python serves the page and its files. On Vercel public/ is handled by the CDN before this code runs,
# so these are only a fallback, and they must never stop the app from starting when public/ is not in the bundle.
@app.get("/")
def index() -> FileResponse:
    page = PUBLIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="public/index.html is not part of this deployment")
    return FileResponse(page)


if (PUBLIC_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", 8080))
    print(f"Job board: http://{host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=False)