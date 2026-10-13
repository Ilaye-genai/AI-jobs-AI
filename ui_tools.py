"""
The tools the agent can call. Each one does a job AND moves the user's page.

    tools = make_tools(send)     # send(message: dict) delivers a message to the user's browser

`send` is passed in so this module knows nothing about LiveKit:
  * voice_agent.py passes a function that publishes on the LiveKit room (topic "ui")
  * agent.py's terminal test passes a function that just prints

The agent only ever sends Job IDs. The page loads the jobs itself from /api/jobs, which reads
scraped_jobs.json, so the website and the vector search can never disagree about a job's content.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from langchain_core.tools import tool

import jobs_data
import vector_store as vs
from job_tools import parse_query

logger = logging.getLogger(__name__)

Send = Callable[[dict], Awaitable[None]]
MAX_RESULTS = 6
SEARCH_CANDIDATES = 10  # fewer candidates to rerank = a faster search
DETAIL_CHARS = 1800  # how much of a description the agent gets to read when it opens a job


async def _safe_send(send: Send, message: dict) -> None:
    """A page update must never crash the conversation."""
    try:
        await send(message)
    except Exception as e:
        logger.warning("Could not update the page (%s): %s", message.get("action"), e)


def _describe(job: dict) -> str:
    """What the agent gets to read about one job (the page shows the full thing to the user)."""
    lines = [
        f"{job['title']} at {job['company']}",
        f"Job ID: {job['id']}",
        f"Pay: {job['pay_label'] or 'not listed'}"
        + (" (the posting says pay is output-based, paid per task, so the hourly range is approximate)" if job["pay_model"] == "per-task" else ""),
        f"Type: {job['employment'] or 'not stated'} | Location: {job['location']}" + (f" | Hours: {job['hours']}" if job["hours"] else ""),
        f"Skills: {', '.join(job['skills'][:8]) or 'not listed'}",
    ]
    if job["has_description"]:
        lines.append("\n" + job["body"][:DETAIL_CHARS])
    else:
        lines.append("\n(The full description could not be scraped for this job.)")
    return "\n".join(lines)


def make_tools(send: Send) -> list:
    @tool
    async def search_jobs(query: str) -> str:
        """Search the job database with a plain-English description and show the matches on the user's page.

        Put everything the user wants into the query: the kind of work, skills, field, and any pay or
        recency requirement, for example "remote Python developer job paying over $50/hour posted this
        week". Returns the matching jobs, best match first, each with its Job ID. Only Job IDs returned
        here are valid.
        """
        await _safe_send(send, {"action": "status", "text": "Searching jobs…", "kind": "search"})
        pq = parse_query(query)
        after = datetime.now(timezone.utc) - timedelta(days=pq.days) if pq.days else None

        def run() -> tuple[list[tuple[dict, dict]], int]:
            """Search, then pair each hit with the website's copy of the job (same file the page reads)."""
            hits = vs.search_jobs(
                pq.semantic, limit=MAX_RESULTS, candidates=SEARCH_CANDIDATES,
                min_pay=pq.min_pay, max_pay=pq.max_pay, posted_after=after,
            )
            pairs = [(h, jobs_data.get_job(h["job_id"])) for h in hits]
            return [(h, j) for h, j in pairs if j], len(hits)

        try:
            found, raw_hits = await asyncio.to_thread(run)
        except Exception as e:
            logger.exception("search failed")
            # Shown on the visitor's page too, so a broken deployment (no database, bad key) is visible.
            await _safe_send(send, {
                "action": "error",
                "text": "The job search failed",
                "hint": "Usually the vector database is not reachable from the agent host: check QDRANT_URL and QDRANT_API_KEY, and that python ingest.py has been run against that database.",
                "detail": f"{type(e).__name__}: {e}"[:500],
            })
            return f"The job search failed: {e}"
        if not found:
            if raw_hits:  # the database returned jobs, but the website does not have them
                await _safe_send(send, {
                    "action": "error",
                    "text": "The search index and the website's job list disagree",
                    "hint": "The search found jobs that are not in this deployment's scraped_jobs.json. Redeploy the website and the agent from the same scraped_jobs.json, or re-run python ingest.py.",
                    "detail": f"{raw_hits} hits, none found in the website's list",
                })
                return ("The search found jobs, but none are in the website's job list. "
                        "The search index and scraped_jobs.json are out of sync; tell the user something is wrong on our side.")
            applied = pq.describe()
            return "No jobs matched." + (f" Filters applied: {applied}. Offer to relax them." if applied else "")

        await _safe_send(send, {"action": "show_results", "ids": [j["id"] for _, j in found]})
        n = len(found)
        await _safe_send(send, {"action": "status", "text": f"Found {n} job{'s' if n != 1 else ''}", "kind": "found"})

        lines = [
            f"{j['id']} | {j['title']} | {j['pay_label'] or 'pay not listed'} | {j['domain']} | "
            f"location: {j['location']} | skills: {', '.join(j['skills'][:5])} | {j['summary'][:160]}"
            for _, j in found
        ]
        return (f"{n} jobs are now shown on the user's page, best match first. "
                "This list is for you only: do NOT read it aloud, just name the best two briefly.\n"
                + "\n".join(lines))

    @tool
    async def show_job(job_id: str) -> str:
        """Open ONE job on the user's page and read its details so you can describe it.

        Use it for the job you recommend, or whenever the user picks or asks about a specific job.
        The job_id must come from search_jobs.
        """
        job = jobs_data.get_job(job_id)
        if not job:
            return "No job with that ID. Only use Job IDs returned by search_jobs."
        await _safe_send(send, {"action": "status", "text": f"Opening {job['title'].strip()}…", "kind": "open"})
        await _safe_send(send, {"action": "show_job", "job_id": job["id"]})
        return _describe(job)

    @tool
    async def highlight_apply(job_id: str) -> str:
        """Make the Apply button pulse on the user's page. Use it only when the user says they want to apply."""
        job = jobs_data.get_job(job_id)
        if not job:
            return "No job with that ID. Only use Job IDs returned by search_jobs."
        await _safe_send(send, {"action": "status", "text": "Pointing at Apply…", "kind": "apply"})
        await _safe_send(send, {"action": "highlight_apply", "job_id": job["id"]})
        return "The Apply button is pulsing. Tell the user to click it themselves; you cannot apply for them."

    @tool
    async def show_all_jobs() -> str:
        """Take the user's page back to the full list of jobs, without your picks."""
        await _safe_send(send, {"action": "status", "text": "Showing all jobs", "kind": "open"})
        await _safe_send(send, {"action": "show_all"})
        return "The page now shows all jobs."

    return [search_jobs, show_job, highlight_apply, show_all_jobs]