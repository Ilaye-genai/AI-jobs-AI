"""
Agent-facing tools. Natural-language string in, readable string out.

    find_jobs("remote medical coding job paying over $50/hour posted this week")
    get_job_details("9461932b-7cfe-41c8-b7db-4d0b0d599187")

Why a parser sits in front of the vector search: embeddings are good at "what kind of job" but
cannot reliably enforce numbers or dates. So amounts ("over $50/hour") and recency ("this week")
are pulled out of the sentence and applied as real Qdrant filters; the rest is the semantic query.
"""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool

import vector_store as vs

MAX_DETAIL_CHARS = 6000

# --------------------------------------------------------------------------------------
# Natural-language query parsing
# --------------------------------------------------------------------------------------
_NUM = r"\d+(?:\.\d+)?(?!\d|\.\d)"  # whole numbers only: never backtrack into "100k" -> "10"
_PAY_RE = re.compile(
    r"(?:(?:paying|pays?|that pays?|which pays?|with (?:a )?(?:pay|rate|salary)(?: of)?)\s+)?"
    r"(?:(?P<qmax>under|below|less than|at most|up to|max(?:imum)?(?: of)?|no more than)|"
    r"(?P<qmin>over|above|at least|min(?:imum)?(?: of)?|more than|greater than|starting (?:at|from)|from|between))?\s*"
    rf"(?:\$\s*(?P<n1>{_NUM})|(?P<n2>{_NUM})(?=\s*(?:/\s*h|an hour|per hour|hourly|dollars)))"
    r"(?!\s*k\b)"
    rf"(?:\s*(?:-|–)\s*\$?\s*{_NUM}|\s+(?:to|and)\s+\$\s*{_NUM})?"  # "$50-80", "$50 to $80"
    r"\s*(?:/\s*h(?:ou)?r?|an hour|per hour|hourly|dollars(?: an hour| per hour)?)?"
    r"(?:\s*\+|\s+(?:or more|and (?:up|above|over)))?",
    re.I,
)

_LEAD = r"(?:\b(?:posted|from|in|within|during)\s+(?:the\s+)?){0,2}"
_RECENCY = [
    (re.compile(_LEAD + r"\b(?:today|last 24 hours|past 24 hours|past day)\b", re.I), 1),
    (re.compile(_LEAD + r"\b(?:this week|past week|last week|last 7 days|past 7 days)\b", re.I), 7),
    (re.compile(_LEAD + r"\b(?:this month|past month|last month|last 30 days|past 30 days)\b", re.I), 30),
    (re.compile(_LEAD + r"\b(?:just posted|newly posted|recently posted|recent(?:ly)?|latest|newest)\b", re.I), 14),
]


@dataclass
class ParsedQuery:
    semantic: str
    min_pay: float | None = None
    max_pay: float | None = None
    days: int | None = None

    def describe(self) -> str:
        bits = []
        if self.min_pay is not None:
            bits.append(f"pay reaching at least ${self.min_pay:g}/hour")
        if self.max_pay is not None:
            bits.append(f"starting pay at most ${self.max_pay:g}/hour")
        if self.days:
            bits.append(f"posted within the last {self.days} day{'s' if self.days != 1 else ''}")
        return ", ".join(bits)


def parse_query(query: str) -> ParsedQuery:
    text = query.strip()
    min_pay = max_pay = None

    def pay_sub(m: re.Match) -> str:
        nonlocal min_pay, max_pay
        amount = float(m.group("n1") or m.group("n2"))
        if m.group("qmax"):
            max_pay = amount if max_pay is None else min(max_pay, amount)
        else:
            min_pay = amount if min_pay is None else max(min_pay, amount)
        return " "

    text = _PAY_RE.sub(pay_sub, text)

    days = None
    for pattern, d in _RECENCY:
        if pattern.search(text):
            days = d if days is None else min(days, d)
            text = pattern.sub(" ", text)

    semantic = re.sub(r"\s+", " ", re.sub(r"\s+([,.;])", r"\1", text)).strip(" ,.;")
    semantic = re.sub(r"^(?:and|or|with|for)\s+", "", semantic, flags=re.I)
    if len(semantic) < 3:  # the whole query was filters, e.g. "jobs over $80/hour"
        semantic = query.strip()
    return ParsedQuery(semantic, min_pay, max_pay, days)


# --------------------------------------------------------------------------------------
# Formatting for the LLM
# --------------------------------------------------------------------------------------
def _format_hit(i: int, h: dict) -> str:
    posted = (h["posted_at"] or "")[:10] or "unknown"
    lines = [
        f"{i}. {h['title']}",
        f"   Job ID: {h['job_id']}",
        f"   Pay: {h['pay'] or 'not listed'} | Field: {(h['domain'] or '').replace('-', ' ')} | "
        f"Level: {h['role_type']} | Posted: {posted}",
        f"   Skills: {', '.join(h['skills'][:8]) or 'not listed'}",
        f"   About: {h['summary']}",
        f"   URL: {h['url']}",
    ]
    return "\n".join(lines)


def _run(query: str, limit: int) -> str:
    pq = parse_query(query)
    after = datetime.now(timezone.utc) - timedelta(days=pq.days) if pq.days else None
    hits = vs.search_jobs(
        pq.semantic, limit=limit, min_pay=pq.min_pay, max_pay=pq.max_pay, posted_after=after
    )
    applied = pq.describe()
    if not hits:
        msg = f"No jobs matched '{query}'."
        if applied:
            msg += f" Filters applied: {applied}. Try relaxing them."
        return msg
    header = f"Found {len(hits)} matching job{'s' if len(hits) != 1 else ''}"
    header += f" (filters: {applied}):" if applied else ":"
    return header + "\n\n" + "\n\n".join(_format_hit(i, h) for i, h in enumerate(hits, 1))


# --------------------------------------------------------------------------------------
# The tools
# --------------------------------------------------------------------------------------
def find_jobs(query: str, limit: int = 5) -> str:
    """Search the job database using a plain-English description of what the person wants.

    Include everything they said in the query: the kind of work, skills, field, and any pay or
    recency requirements, e.g. "remote medical coding job paying over $50/hour" or
    "Python developer jobs posted this week". Returns up to `limit` matching jobs with their
    Job ID, pay, skills, a short summary and URL. Only Job IDs returned here are valid.
    """
    try:
        return _run(query, max(1, min(int(limit), 10)))
    except Exception as e:
        return f"Job search failed: {e}"


def get_job_details(job_id: str) -> str:
    """Get the full description of ONE job, using a Job ID returned by find_jobs.

    Use this when the person asks for more detail about a specific job (responsibilities,
    requirements, location, what the work involves).
    """
    try:
        p = vs.get_job(job_id)
    except Exception as e:
        return f"Could not load the job: {e}"
    if not p:
        return "No job found with that ID. Only use Job IDs returned by find_jobs."
    desc = vs.clean_description(p.get("description"), p) or "(No detailed description was available for this job.)"
    if len(desc) > MAX_DETAIL_CHARS:
        desc = desc[:MAX_DETAIL_CHARS].rstrip() + "\n[...description truncated...]"
    return (
        f"{p['title']}\n"
        f"Job ID: {p['job_id']}\n"
        f"Pay: {vs.pay_text(p) or 'not listed'} | Level: {p.get('role_type')} | "
        f"Field: {(p.get('domain') or '').replace('-', ' ')} | Posted: {(p.get('posted_at') or '')[:10]}\n"
        f"Skills: {', '.join(p.get('skills') or []) or 'not listed'}\n"
        f"URL: {p.get('url')}\n\n{desc}"
    )


# Hand these to your LangGraph agent (e.g. create_react_agent(llm, TOOLS) or llm.bind_tools(TOOLS)).
TOOLS = [tool(find_jobs), tool(get_job_details)]
import time
import re

if __name__ == "__main__":
    start = time.time()
    results = find_jobs("remote coding job paying over $50/hour posted this week")

    # Pull the Job IDs out of the search output and fetch each full description
    job_ids = re.findall(r"Job ID: (\S+)", results)
    full = [get_job_details(jid) for jid in job_ids]

    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(results)
        f.write("\n\n" + "=" * 80 + "\nFULL DETAILS\n" + "=" * 80 + "\n\n")
        f.write(("\n\n" + "-" * 80 + "\n\n").join(full))

    print("Results saved to results.txt")
    print(f"Total time taken: {time.time() - start:.2f} seconds")