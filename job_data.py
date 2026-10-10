"""
The single reader of scraped_jobs.json for the website.

    import jobs_data
    jobs_data.search(q="python", category="Engineering & Data", min_pay=50)
    jobs_data.get_job("90f04fad-...")

Why this exists: the frontend and the vector DB must never disagree about which jobs exist.
Both are fed from scraped_jobs.json (ingest.py writes the vectors, this module serves the page),
and the page is joined to Qdrant results purely by job_id. Re-scrape, and the site follows:
the file's modified time is checked on every call and the data is re-read only when it changed.
No frontend code depends on which jobs, domains or skills happen to exist.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from pathlib import Path

from classify import enrich

JOBS_FILE = Path(os.getenv("JOBS_FILE") or Path(__file__).with_name("scraped_jobs.json"))

# Fields kept out of list responses (they are big, or only used for server-side filtering).
_HEAVY = {"body", "search_text"}

_lock = threading.Lock()
_state: dict = {"mtime": None, "jobs": [], "by_id": {}, "facets": {}}


def _facets(jobs: list[dict]) -> dict:
    def counted(key: str) -> list[dict]:
        c = Counter(j[key] for j in jobs if j.get(key))
        return [{"value": v, "count": n} for v, n in c.most_common()]

    skills = Counter(s for j in jobs for s in j["skills"] if len(s) <= 28)
    pays = [j["pay_max"] for j in jobs if j.get("pay_max") is not None]
    return {
        "category": counted("category"),
        "domain": counted("domain"),
        "level": counted("level"),
        "employment": counted("employment"),
        "company": counted("company"),
        "skills": [{"value": v, "count": n} for v, n in skills.most_common(40)],
        "pay_max": max(pays) if pays else None,
    }


def _reload_if_changed() -> None:
    """Re-read the file if a scrape rewrote it. Keeps serving the old data on a half-written file."""
    try:
        mtime = JOBS_FILE.stat().st_mtime_ns
    except FileNotFoundError:
        return
    if mtime == _state["mtime"]:
        return
    try:
        raw = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return  # the scraper is mid-write; try again on the next call
    by_id: dict[str, dict] = {}
    for item in raw:
        if not item.get("job_id") or not item.get("title"):
            continue
        job = enrich(item)
        by_id[job["id"].lower()] = job  # a repeated id: last one wins
    jobs = sorted(by_id.values(), key=lambda j: (j["posted"] or "", j["title"]), reverse=True)
    _state.update(mtime=mtime, jobs=jobs, by_id=by_id, facets=_facets(jobs))


def _ensure() -> None:
    with _lock:
        _reload_if_changed()


def version() -> str:
    """Changes whenever scraped_jobs.json changes. The page polls this to offer a refresh."""
    _ensure()
    return str(_state["mtime"] or 0)


def facets() -> dict:
    _ensure()
    return _state["facets"]


def get_job(job_id: str) -> dict | None:
    """One full job (including the description body), or None."""
    _ensure()
    return _state["by_id"].get((job_id or "").strip().lower())


def to_list_item(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in _HEAVY}


def get_many(ids: list[str]) -> list[dict]:
    """List items for these ids, in the order given (the agent's ranking). Unknown ids are skipped."""
    _ensure()
    found = (_state["by_id"].get(i.strip().lower()) for i in ids)
    return [to_list_item(j) for j in found if j]


def search(
    q: str = "",
    category: str | None = None,
    domain: str | None = None,
    level: str | None = None,
    employment: str | None = None,
    company: str | None = None,
    skill: str | None = None,
    min_pay: float | None = None,
    sort: str = "new",
    limit: int = 60,
    offset: int = 0,
) -> tuple[int, list[dict]]:
    """Filter and sort the board. Returns (total matches, one page of list items)."""
    _ensure()
    words = [w for w in (q or "").lower().split() if w]

    def keep(j: dict) -> bool:
        if category and j["category"] != category:
            return False
        if domain and j["domain"] != domain:
            return False
        if level and j["level"] != level:
            return False
        if employment and j["employment"] != employment:
            return False
        if company and j["company"] != company:
            return False
        if skill and skill.lower() not in (s.lower() for s in j["skills"]):
            return False
        if min_pay is not None and (j["pay_max"] is None or j["pay_max"] < min_pay):
            return False
        return all(w in j["search_text"] for w in words)

    matches = [j for j in _state["jobs"] if keep(j)]
    if sort == "pay":
        matches.sort(key=lambda j: (j["pay_max"] is not None, j["pay_max"] or 0), reverse=True)
    limit = max(1, min(int(limit), 200))
    page = matches[max(0, offset): max(0, offset) + limit]
    return len(matches), [to_list_item(j) for j in page]