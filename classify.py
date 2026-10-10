"""
Turn one raw scraped job into the clean, structured shape the website uses.

Deliberately has NO heavy imports: the web server imports this, and it must start instantly
(vector_store.py pulls in fastembed and qdrant, which the website does not need).

    from classify import enrich
    job = enrich(raw_job)      # raw dict from scraped_jobs.json -> website-ready dict

Most structure already exists in the scrape (domain, role_type, pay_min/max, skills, posted).
This module adds what the scrape does not give you directly:
  * category      - one fixed taxonomy across ALL job sites, so filters stay consistent when you
                    add Mercor and others whose own "domain" names differ
  * pay_model     - "per-task" when the posting says pay is output-based, else "hourly"
  * location, employment, hours - pulled out of the description text
  * body, summary - the description minus the boilerplate the page already shows elsewhere
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

# --------------------------------------------------------------------------------------
# Category taxonomy. Order matters: the first rule that matches the domain + title wins.
# --------------------------------------------------------------------------------------
CATEGORY_RULES: list[tuple[str, re.Pattern]] = [
    ("Healthcare & Life Sciences", re.compile(
        r"medic|physician|hospitalist|clinical|nurs|pharm|health|biolog|dental|therap|surgeon|psychiatr|radiolog", re.I)),
    ("Legal", re.compile(r"legal|\blaw\b|lawyer|attorney|paralegal|litigation", re.I)),
    ("Finance & Business", re.compile(
        r"financ|quant|account|bank|invest|econom|trading|audit|\btax\b|actuar|insurance|consult", re.I)),
    ("Engineering & Data", re.compile(
        r"software|engineer|developer|programm|machine learning|\bml\b|\bai\b|data scien|data engineer|"
        r"devops|backend|frontend|full.?stack|mobile|security|cloud|\bqa\b|coding|python|rust", re.I)),
    ("Science & Math", re.compile(r"physic|chemi|\bmath|statistic|scien|astronom|geolog|research", re.I)),
    ("Languages & Writing", re.compile(
        r"translat|linguist|language|writer|writing|editor|copy|journalis|content|transcri", re.I)),
    ("Creative & Design", re.compile(r"design|\bart\b|artist|music|video|animat|creative|photo|illustrat", re.I)),
    ("Operations & Management", re.compile(
        r"manager|operations|project|recruit|support|human data|coordinator|analyst|generalist|admin", re.I)),
]
DEFAULT_CATEGORY = "Other"

_UNIT_LABEL = {"hour": "/hr", "year": "/yr", "month": "/mo", "week": "/wk", "day": "/day"}

_FIELD_KEYS = r"Role Type|Job Type|Location|Schedule|Job Summary|Role Title|Job Title|Duration|Hours"
_PAY_LINE = re.compile(r"^\$[\d,.]+(\s*-\s*\$?[\d,.]+)?\s*/?\s*\w*$")
_META_LINE = re.compile(r"^\*\*(?:role title|role type|job title|job type|location|schedule):\*\*", re.I)
_PER_TASK = re.compile(r"output[- ]based|paid per task|per[- ]task|per completed task", re.I)


def categorize(domain: str | None, title: str | None) -> str:
    text = f"{(domain or '').replace('-', ' ')} {title or ''}"
    for name, pattern in CATEGORY_RULES:
        if pattern.search(text):
            return name
    return DEFAULT_CATEGORY


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _fmt_money(x: float) -> str:
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"


def pay_label(lo, hi, unit: str | None) -> str | None:
    if lo is None and hi is None:
        return None
    suffix = _UNIT_LABEL.get((unit or "hour").lower(), f"/{unit}")
    if lo is None:
        return f"up to ${_fmt_money(hi)}{suffix}"
    if hi is None or hi == lo:
        return f"${_fmt_money(lo)}{suffix}"
    return f"${_fmt_money(lo)}–{_fmt_money(hi)}{suffix}"


def _iso(ts) -> str | None:
    """'2026-10-10 01:44:59' -> '2026-10-10T01:44:59Z' (the scrape is UTC, same as vector_store)."""
    if not ts:
        return None
    s = str(ts).strip().replace(" ", "T")
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_skills(skills) -> list[str]:
    out, seen = [], set()
    for s in skills or []:
        s = re.sub(r"\s+", " ", str(s)).strip()
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s if any(c.isupper() for c in s) else s[:1].upper() + s[1:])
    return out


def _field(desc: str, key: str) -> str | None:
    """Read 'Location: Remote' whether it is **bold** on its own line or inline in one paragraph."""
    m = re.search(
        rf"(?:\*\*)?{key}(?::\*\*|:)\s*(.+?)(?=\s+(?:\*\*)?(?:{_FIELD_KEYS})(?::\*\*|:)|\n|$)",
        desc, re.I,
    )
    return m.group(1).strip(" *") if m else None


def _employment(role_type: str | None, job_type: str | None) -> str | None:
    value = (job_type or role_type or "").lower()
    if not value:
        return None
    if "full" in value:
        return "Full-time"
    if "part" in value:
        return "Part-time"
    if "contract" in value or "freelance" in value:
        return "Contractor"
    return value.strip().title()[:30] or None


def _location(desc: str) -> str:
    """'Location: Remote In this role...' (inline, no line break) must give just 'Remote'."""
    value = _field(desc, "Location")
    if not value:
        return "Remote"
    m = re.match(r"(remote|hybrid|on-?site)\b(\s*\([^)]{1,40}\))?", value, re.I)
    if m:
        return m.group(0).strip()
    if len(value) > 40:  # a real place name is short; anything longer ran into the next sentence
        value = value[:40].rsplit(" ", 1)[0]
    return value


def _hours(desc: str) -> str | None:
    m = re.search(r"~?\s*(\d{1,2})\s*(?:-\s*\d{1,2}\s*)?(?:hrs?|hours)\s*(?:a|per|/)\s*week", desc, re.I)
    return f"~{m.group(1)} hrs/week" if m else None


def clean_body(desc: str | None, title: str | None, skills: list[str]) -> str:
    """
    The description with what the page already shows elsewhere removed: the repeated title,
    the pay line, the 'Required Skills' list, and the Role Title / Role Type / Location lines.
    Markdown is kept (the frontend renders it safely).
    """
    if not desc:
        return ""
    title_l = (title or "").strip().lower()
    skill_set = {s.lower() for s in skills}
    out: list[str] = []
    in_skills = False
    for line in desc.splitlines():
        s = line.strip()
        low = s.lower()
        if not s:
            out.append("")
            continue
        if title_l and re.sub(r"^#+\s*", "", low).strip() == title_l:
            continue
        if low.startswith("$") and _PAY_LINE.match(re.sub(r"pay$", "", low).strip()):
            continue
        if re.match(r"^#{1,6}\s*required skills\s*$", low):
            in_skills = True
            continue
        if in_skills:
            if low in skill_set:
                continue
            in_skills = False
        if _META_LINE.match(s):
            continue
        out.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _summary(body: str, n: int = 240) -> str:
    """First real paragraph of the body, plain text, cut on a word boundary."""
    for block in re.split(r"\n\s*\n", body):
        text = re.sub(r"\*\*|__|^#+\s*", "", block).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) < 40 or text.endswith(":"):
            continue
        if len(text) <= n:
            return text
        cut = text[:n]
        return (cut[: cut.rfind(" ")] if " " in cut else cut) + "…"
    return ""


# --------------------------------------------------------------------------------------
# The one public function
# --------------------------------------------------------------------------------------
def enrich(job: dict) -> dict:
    """Raw scraped job -> website-ready job. Never raises on missing fields."""
    desc = job.get("description") or ""
    skills = _clean_skills(job.get("skills"))
    lo, hi = job.get("pay_min"), job.get("pay_max")
    domain = (job.get("domain") or "").replace("-", " ").strip()
    body = clean_body(desc, job.get("title"), skills)
    summary = _summary(body)
    level = (job.get("role_type") or "").strip().title() or None
    # The scrape's own job_type is just "REGULAR"; the real answer is in the description text.
    employment = _employment(_field(desc, "Role Type"), _field(desc, "Job Type"))
    location = _location(desc)
    category = categorize(domain, job.get("title"))

    search_text = " ".join(
        [job.get("title") or "", job.get("company") or "", domain, category, summary, " ".join(skills)]
    ).lower()

    return {
        "id": job.get("job_id"),
        "url": job.get("url"),
        "apply_url": job.get("apply_url") or job.get("url"),
        "title": job.get("title") or "Untitled job",
        "company": job.get("company") or "Unknown",
        "domain": domain or "other",
        "category": category,
        "level": level,
        "employment": employment,
        "location": location,
        "hours": _hours(desc),
        "pay_min": lo,
        "pay_max": hi,
        "pay_unit": job.get("pay_unit") or "hour",
        "pay_label": pay_label(lo, hi, job.get("pay_unit")),
        "pay_model": "per-task" if _PER_TASK.search(desc) else "hourly",
        "skills": skills,
        "posted": _iso(job.get("posted")),
        "summary": summary,
        "body": body,
        "has_description": bool(desc),
        "search_text": search_text,
    }