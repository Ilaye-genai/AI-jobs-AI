"""Mercor job collector.

Fetches Mercor listing metadata from its listing endpoint, opens each listing page,
extracts the rendered page as cleaned Markdown, and returns records in the common
schema used by web_scraper.py and ingest.py.

Examples:
    python mercor.py --limit 5
    python mercor.py --limit 5 --full
    python mercor.py --concurrency 4

This module does not write scraped_jobs.json itself. ingest.py merges Mercor and
Micro1 records and writes the shared snapshot so one site's scrape cannot overwrite
 the other site's jobs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from classify import clean_body

LIST_URL = "https://work.mercor.com/explore"
MERCOR_API_URL = "https://aws.api.mercor.com/work/listings-explore-page"
JOB_URL = "https://work.mercor.com/jobs/{}"

MIN_CHARS = 200
MIN_BODY_CHARS = 150
PREVIEW_CHARS = 500
DEFAULT_CONCURRENCY = 4
RETRY_CONCURRENCY = 2

_crawler: AsyncWebCrawler | None = None
_lock: asyncio.Lock | None = None

# Query strings are usually analytics/tracking parameters. Preserve query strings for
# sites where they identify the actual destination, such as YouTube video links.
KEEP_QUERY_HOSTS = ("youtube.com", "youtu.be")

# These are section headings / labels, not ordinary text within a job description.
_DROP_HEADING_PATTERNS = (
    re.compile(r"^about\s+mercor\b", re.IGNORECASE),
    re.compile(r"^application\b", re.IGNORECASE),
)
_DROP_TO_DIVIDER_PATTERNS = (
    re.compile(r"^similar\s+opportunities\b", re.IGNORECASE),
    re.compile(r"^earn\s+up\s+to\b", re.IGNORECASE),
)
_DROP_TO_END_PATTERNS = (
    re.compile(r"^one\s+interview,\s*real\s+results\b", re.IGNORECASE),
)

_DIVIDER_RE = re.compile(r"^\s*(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})\s*$")
_HEADING_RE = re.compile(r"^\s*(#{1,6})[ \t]+(.+?)\s*$")
_BOLD_LABEL_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*:?[ \t]*$")


def _heading_parts(line: str) -> tuple[int, str] | None:
    match = _HEADING_RE.match(line)
    if not match:
        return None
    text = match.group(2).strip()
    # Allow headings like ## **About Mercor** or ## **What you'll do:**.
    text = re.sub(r"^\*\*(.*?)\*\*$", r"\1", text).strip()
    return len(match.group(1)), text


def _drop_sections(md: str) -> str:
    """Remove irrelevant sections without deleting the next real section.

    Heading sections (Application / About Mercor) end when a heading at the same
    or a higher level appears. Similar-opportunities and referral sections are
    removed through their next horizontal divider.
    """
    output: list[str] = []
    skip_heading_level: int | None = None
    skip_until_divider = False
    skip_bold_section = False
    skip_to_end = False

    for line in md.splitlines():
        if skip_to_end:
            continue
        heading = _heading_parts(line)
        bold_match = _BOLD_LABEL_RE.match(line)
        bold_label = bold_match.group(1).strip() if bold_match else None

        if skip_until_divider:
            if _DIVIDER_RE.match(line):
                skip_until_divider = False
            continue

        if skip_heading_level is not None:
            if heading and heading[0] <= skip_heading_level:
                skip_heading_level = None
            else:
                continue

        if skip_bold_section:
            if heading or bold_label:
                skip_bold_section = False
            else:
                continue

        label = heading[1] if heading else bold_label
        if label is None:
            # Mercor sometimes emits these section titles without Markdown hashes.
            stripped = line.strip()
            if re.match(r"(?i)^similar\s+opportunities\b", stripped):
                label = stripped
            elif re.match(r"(?i)^earn\s+up\s+to\b", stripped):
                label = stripped
            elif re.match(r"(?i)^one\s+interview,\s*real\s+results\b", stripped):
                label = stripped

        if label:
            if any(pattern.match(label) for pattern in _DROP_TO_END_PATTERNS):
                skip_to_end = True
                continue

            if any(pattern.match(label) for pattern in _DROP_TO_DIVIDER_PATTERNS):
                skip_until_divider = True
                continue

            if any(pattern.match(label) for pattern in _DROP_HEADING_PATTERNS):
                if heading:
                    skip_heading_level = heading[0]
                else:
                    skip_bold_section = True
                continue

        output.append(line)

    return "\n".join(output)


def _clean_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.netloc.lower()
    if any(host == h or host.endswith("." + h) for h in KEEP_QUERY_HOSTS):
        return url
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def clean_markdown(md: str) -> str:
    """Clean rendered Mercor page Markdown, retaining job-specific information."""
    # Remove images (logos and tracking pixels).
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)

    # Remove company/application/related-job/referral sections.
    md = _drop_sections(md)

    # Remove standalone page chrome and Mercor attribution, not job text.
    line_patterns = (
        r"^\s*\[←\s*View all opportunities\]\([^)]*\)\s*$",
        r"^\s*Posted by Mercor\s*$",
        r"^\s*\[mercor\.com\]\(https?://(?:www\.)?mercor\.com/?(?:[?#][^)]*)?\)\s*$",
        r"^\s*Activate my referral link\s*$",
        r"^\s*Don't know who to refer\?.*$",
        r"^\s*Apply now\s*$",
    )
    for pattern in line_patterns:
        md = re.sub(pattern, "", md, flags=re.MULTILINE | re.IGNORECASE)

    # Strip tracking parameters from links while preserving YouTube query strings.
    md = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        lambda match: f"[{match.group(1)}]({_clean_url(match.group(2))})",
        md,
    )

    # Unwrap italics around entire lines and tidy list indentation / spacing.
    md = re.sub(r"^_(.+)_$", r"\1", md, flags=re.MULTILINE)
    md = re.sub(r"^\s{2,}(\d+\.)", r"\1", md, flags=re.MULTILINE)
    md = re.sub(r"[ \t]+$", "", md, flags=re.MULTILINE)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


async def get_crawler() -> AsyncWebCrawler:
    global _crawler, _lock
    if _lock is None:
        _lock = asyncio.Lock()
    if _crawler is None:
        async with _lock:
            if _crawler is None:
                browser_config = BrowserConfig(
                    headless=True,
                    extra_args=["--disable-gpu", "--disable-software-rasterizer"],
                )
                _crawler = AsyncWebCrawler(config=browser_config)
                await _crawler.start()
    return _crawler


async def close_crawler() -> None:
    """Close Crawl4AI resources and reset module globals."""
    global _crawler, _lock
    crawler = _crawler
    _crawler = None
    try:
        if crawler is not None:
            await crawler.close()
            # Give pending shutdown callbacks a chance to run before asyncio.run exits.
            await asyncio.sleep(0)
    finally:
        _lock = None


def _page_config(wait_until: str, delay: float, timeout_ms: int) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_until=wait_until,
        page_timeout=timeout_ms,
        delay_before_return_html=delay,
        remove_overlay_elements=True,
        excluded_tags=["script", "style", "nav", "footer", "svg", "iframe", "noscript"],
        exclude_all_images=True,
        exclude_social_media_links=True,
        verbose=False,
        markdown_generator=DefaultMarkdownGenerator(
            options={"ignore_images": True, "skip_internal_links": True}
        ),
    )


_ATTEMPTS = [
    ("networkidle", 2.0, 30_000),
    ("domcontentloaded", 6.0, 45_000),
]
_PATIENT_ATTEMPTS = [
    ("domcontentloaded", 12.0, 60_000),
    ("domcontentloaded", 30.0, 90_000),
]


def _job_quality(text: str, job: dict) -> int:
    skills = [re.sub(r"\s+", " ", str(skill)).strip() for skill in job.get("skills") or []]
    return len(clean_body(text, job.get("title"), skills))


async def _fetch_page_text(
    url: str,
    quality: Callable[[str], int] = len,
    enough: int = MIN_CHARS,
    accept_partial: bool = False,
    attempts: list[tuple[str, float, int]] | None = None,
) -> str:
    """Fetch one page, retrying until enough useful text renders."""
    crawler = await get_crawler()
    best_text = ""
    best_quality = 0
    last_error = "unknown error"

    for wait_until, delay, timeout in attempts or _ATTEMPTS:
        try:
            result = await crawler.arun(
                url=url,
                config=_page_config(wait_until, delay, timeout),
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue

        if not result.success:
            last_error = result.error_message or "crawl failed"
            continue

        text = clean_markdown(str(result.markdown or ""))
        quality_score = quality(text)
        if quality_score >= enough:
            return text

        if quality_score > best_quality:
            best_text, best_quality = text, quality_score

        preview = re.sub(r"\s+", " ", text)[:120]
        if quality_score == len(text):
            last_error = f"page rendered only {len(text)} chars: {preview!r}"
        else:
            last_error = (
                f"page had {len(text)} chars but only {quality_score} of description text "
                f"(page said: {preview!r})"
            )

    if accept_partial and best_quality > 0:
        return best_text
    raise RuntimeError(last_error)


async def web_crawler(url: str) -> str:
    """Scrape a URL to cleaned Markdown."""
    try:
        return await _fetch_page_text(url)
    except Exception as exc:
        return f"Scrape error: {exc}"


def _http_json(url: str) -> Any:
    """Perform a public JSON GET request; kept synchronous for asyncio.to_thread."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Origin": "https://work.mercor.com",
            "Referer": "https://work.mercor.com/",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0 Safari/537.36"
            ),
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Mercor listing API returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Mercor listing API: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Mercor listing API returned invalid JSON: {exc}") from exc


def fetch_mercor_listings() -> list[dict]:
    """Fetch listing records from Mercor's listing endpoint."""
    payload = _http_json(MERCOR_API_URL)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected Mercor response type: {type(payload).__name__}")

    root = payload.get("response", payload)
    listings = root.get("listings") if isinstance(root, dict) else None
    if not isinstance(listings, list):
        keys = list(payload.keys())
        raise RuntimeError(f"Unexpected Mercor response structure; top-level keys: {keys}")

    return [listing for listing in listings if isinstance(listing, dict)]


def _first_value(record: dict, keys: tuple[str, ...], default=None):
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def _normalise_skills(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;\n]", value) if part.strip()]
    if isinstance(value, dict):
        return [str(item).strip() for item in value.values() if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _number(value: Any):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def job_id(job: dict) -> str | None:
    """Return the platform's native listing ID (it is not necessarily a UUID)."""
    value = _first_value(
        job,
        ("listingId", "listing_id", "job_id", "jobId", "id", "_id", "uid"),
    )
    return str(value).strip() if value is not None else None


def normalize(listing: dict) -> dict:
    """Map Mercor's listing schema to the shared job-record schema."""
    native_id = job_id(listing)
    listing_url = _first_value(listing, ("listingUrl", "url", "jobUrl", "job_url"))
    if listing_url and str(listing_url).startswith("/"):
        listing_url = "https://work.mercor.com" + str(listing_url)
    if not listing_url and native_id:
        listing_url = JOB_URL.format(native_id)

    rate_min = _number(_first_value(listing, ("rateMin", "rate_min", "pay_min")))
    rate_max = _number(_first_value(listing, ("rateMax", "rate_max", "pay_max")))
    rate_obj = _first_value(listing, ("ideal_hourly_rate", "rate"), {})
    if isinstance(rate_obj, dict):
        if rate_min is None:
            rate_min = _number(_first_value(rate_obj, ("min", "minimum", "minRate")))
        if rate_max is None:
            rate_max = _number(_first_value(rate_obj, ("max", "maximum", "maxRate")))

    pay_unit = _first_value(
        listing,
        ("payRateFrequency", "rateFrequency", "pay_unit", "payUnit"),
        "unknown",
    )
    hours_per_week = _first_value(listing, ("hoursPerWeek", "hours_per_week"))

    return {
        # Kept as the native ID here for URL creation and source-level deduplication.
        # ingest.py converts non-UUID IDs to stable UUIDs for Qdrant and saves the
        # native identifier separately as source_job_id.
        "job_id": native_id,
        "source_job_id": native_id,
        "source": "mercor",
        "url": str(listing_url) if listing_url else None,
        "title": _first_value(listing, ("title", "job_name", "jobTitle", "name")),
        "company": _first_value(
            listing,
            ("companyName", "company_name", "company", "organizationName", "organization_name"),
        ),
        "posted": _first_value(listing, ("postedAt", "date_posted", "posted_at", "createdAt", "created_at")),
        "pay_min": rate_min,
        "pay_max": rate_max,
        "pay_unit": str(pay_unit),
        "skills": _normalise_skills(_first_value(listing, ("skills", "requiredSkills", "required_skills"))),
        "role_type": _first_value(listing, ("roleType", "role_type", "commitment", "payRateFrequency")),
        "domain": _first_value(listing, ("listingDomain", "domain", "domainSlug", "domain_slug")),
        "job_type": _first_value(listing, ("jobType", "job_type", "commitment", "payRateFrequency")),
        "apply_url": _first_value(listing, ("applyUrl", "apply_url", "applicationUrl"), str(listing_url) if listing_url else None),
        # Per your requirement, only hoursPerWeek is retained in extra.
        "extra": {"hoursPerWeek": hours_per_week} if hours_per_week is not None else {},
        "description": None,
        "scrape_error": None,
    }


async def enrich_jobs(
    jobs: list[dict],
    concurrency: int = DEFAULT_CONCURRENCY,
    attempts: list[tuple[str, float, int]] | None = None,
) -> None:
    """Scrape descriptions into the supplied records in place."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    semaphore = asyncio.Semaphore(concurrency)
    done = 0
    total = sum(1 for job in jobs if job.get("url"))

    async def scrape_one(job: dict) -> None:
        nonlocal done
        async with semaphore:
            try:
                job["description"] = await _fetch_page_text(
                    job["url"],
                    quality=lambda text: _job_quality(text, job),
                    enough=MIN_BODY_CHARS,
                    accept_partial=True,
                    attempts=attempts,
                )
                job["scrape_error"] = None
            except Exception as exc:
                job["scrape_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                done += 1
                status = "ERR" if job.get("scrape_error") else "ok "
                print(f"  [{done}/{total}] {status} {job.get('title') or job.get('job_id')}")

    await asyncio.gather(*(scrape_one(job) for job in jobs if job.get("url")))


def needs_retry(job: dict) -> bool:
    return bool(job.get("url")) and _job_quality(job.get("description") or "", job) == 0


async def retry_missing(jobs: list[dict], concurrency: int = RETRY_CONCURRENCY) -> int:
    """Re-scrape only records that still have no usable description."""
    todo = [job for job in jobs if needs_retry(job)]
    if not todo:
        print("Every Mercor job already has a usable description.")
        return 0

    print(f"Retrying {len(todo)} of {len(jobs)} Mercor jobs (concurrency={concurrency}) ...")
    try:
        await enrich_jobs(todo, concurrency, attempts=_PATIENT_ATTEMPTS)
    finally:
        await close_crawler()
    return sum(1 for job in todo if not needs_retry(job))


async def collect_jobs(
    limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict]:
    """Return normalized Mercor job records with scraped descriptions."""
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    try:
        raw = await asyncio.to_thread(fetch_mercor_listings)
        print(f"Mercor listing API: {len(raw)} jobs")

        jobs = [normalize(listing) for listing in raw]
        # Deduplicate by native ID, preserving API order.
        unique: list[dict] = []
        seen: set[str] = set()
        for job in jobs:
            key = str(job.get("source_job_id") or job.get("url") or job.get("title"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(job)
        jobs = unique

        if limit is not None:
            jobs = jobs[:limit]
            print(f"Limiting Mercor to the first {len(jobs)} jobs")

        print(f"Scraping Mercor job pages (concurrency={concurrency}) ...")
        await enrich_jobs(jobs, concurrency)
        return jobs
    finally:
        await close_crawler()


def _pay(job: dict) -> str:
    low, high = job.get("pay_min"), job.get("pay_max")
    if low is None:
        return "n/a"

    unit = str(job.get("pay_unit") or "").strip().lower()
    suffixes = {
        "hour": "/h", "hourly": "/h", "per_hour": "/h",
        "year": "/year", "yearly": "/year", "annual": "/year",
        "month": "/month", "monthly": "/month",
        "week": "/week", "weekly": "/week",
        "task": "/task", "per_task": "/task",
        "one-time": "/one-time", "one_time": "/one-time",
    }
    suffix = suffixes.get(unit, f"/{unit}" if unit and unit != "unknown" else "")
    if high is None or high == low:
        return f"${low:g}{suffix}"
    return f"${low:g}-{high:g}{suffix}"


def print_jobs(jobs: list[dict], full: bool = False) -> None:
    print("\n" + "=" * 80)
    for index, job in enumerate(jobs, 1):
        print(f"[{index}] {job.get('title')}  ({job.get('company')})")
        print(f"    url     : {job.get('url')}")
        print(f"    pay     : {_pay(job)}")
        print(f"    posted  : {job.get('posted')}")
        print(f"    type    : {job.get('job_type')} / {job.get('domain')} / {job.get('role_type')}")
        print(f"    skills  : {', '.join(job.get('skills') or [])}")
        if job.get("extra"):
            print(f"    extra   : {json.dumps(job['extra'], ensure_ascii=False)}")
        description = job.get("description")
        if description:
            shown = description if full or len(description) <= PREVIEW_CHARS else description[:PREVIEW_CHARS].rstrip() + " …"
            print(f"    content : ({len(description)} chars)")
            print("      " + shown.replace("\n", "\n      "))
        else:
            print(f"    content : NOT SCRAPED -> {job.get('scrape_error')}")
        print("-" * 80)

    successful = sum(1 for job in jobs if job.get("description"))
    failed = [job for job in jobs if not job.get("description")]
    print(f"Total: {len(jobs)} jobs | content scraped: {successful} | failed: {len(failed)}")
    for job in failed:
        print(f"  FAILED {job.get('url')} ({job.get('scrape_error')})")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape public Mercor job listings.")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N jobs")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--full", action="store_true", help="print full descriptions")
    args = parser.parse_args()

    jobs = await collect_jobs(limit=args.limit, concurrency=args.concurrency)
    print_jobs(jobs, full=args.full)


if __name__ == "__main__":
    asyncio.run(main())
