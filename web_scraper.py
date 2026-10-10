"""
micro1 job collector.

1. Opens the opportunities page once and captures the job-list API call the page makes.
2. Replays it to get every job's metadata (all ~270).
3. Visits each job's page, extracts the description as clean markdown.
4. Merges both into one record per job, ready for an AI recommender.

Usage:
    python web_scraper.py                 # all jobs, short description preview
    python web_scraper.py --limit 5       # test on the first 5 jobs
    python web_scraper.py --full          # print the full description of every job
    python web_scraper.py --concurrency 6
"""
import argparse
import asyncio
import json
import re
import urllib.error
import urllib.request
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langchain_core.tools import tool  # add @tool above web_crawler when you register it with the agent
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

LIST_URL = "https://www.micro1.ai/experts/opportunities"
API_HINT = "/api/v1/job/portal"
JOB_URL = "https://jobs.micro1.ai/post/{}"

MIN_CHARS = 200          # a scraped page shorter than this is treated as "not rendered" and retried
PREVIEW_CHARS = 500      # description preview length when printing (use --full for everything)
DEFAULT_CONCURRENCY = 4  # parallel job pages; raise carefully, each one is a browser tab

_crawler: AsyncWebCrawler | None = None
_lock: asyncio.Lock | None = None

# Headings whose whole section gets dropped. Matched against the heading text.
# "About\b" removes any "About ..." section; use r"About micro1" to be stricter.
SECTIONS_TO_DROP = [r"About\b"]

# Hosts where the query string is part of the link and must be kept.
KEEP_QUERY_HOSTS = ("youtube.com", "youtu.be")


# ----------------------------------------------------------------------------
# Markdown cleaning (job page content)
# ----------------------------------------------------------------------------
def _drop_sections(md: str) -> str:
    """Remove a heading and its content up to the next heading or bold label line."""
    for title in SECTIONS_TO_DROP:
        pattern = rf"^#{{1,6}}[ \t]*{title}.*?(?=^#{{1,6}}[ \t]|^\*\*|\Z)"
        md = re.sub(pattern, "", md, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    return md


def _clean_url(url: str) -> str:
    parts = urlsplit(url)
    if any(h in parts.netloc.lower() for h in KEEP_QUERY_HOSTS):
        return url
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def clean_markdown(md: str) -> str:
    # 1. Remove images (logos, tracking pixels)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)

    # 2. Remove irrelevant sections such as "About <company>"
    md = _drop_sections(md)

    # 3. Strip tracking params from link URLs (keeps YouTube's ?v=)
    md = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        lambda m: f"[{m.group(1)}]({_clean_url(m.group(2))})",
        md,
    )

    # 4. Unwrap italics wrapped around whole lines
    md = re.sub(r"^_(.+)_$", r"\1", md, flags=re.MULTILINE)

    # 5. Remove indentation left over from nested HTML lists
    md = re.sub(r"^\s{2,}(\d+\.)", r"\1", md, flags=re.MULTILINE)

    # 6. Trim trailing whitespace and collapse runs of blank lines
    md = re.sub(r"[ \t]+$", "", md, flags=re.MULTILINE)
    md = re.sub(r"\n{3,}", "\n\n", md)

    return md.strip()


# ----------------------------------------------------------------------------
# Crawler
# ----------------------------------------------------------------------------
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
    global _crawler
    if _crawler is not None:
        await _crawler.close()
        _crawler = None


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


# First try: wait for the network to settle. Second try: don't wait for idle (analytics pings can
# keep a page "busy" forever) but give the JS a few seconds to render instead.
_ATTEMPTS = [("networkidle", 2.0, 30_000), ("domcontentloaded", 6.0, 45_000)]


async def _fetch_page_text(url: str) -> str:
    """Scrape one page to cleaned markdown. Raises RuntimeError if it can't get real content."""
    crawler = await get_crawler()
    last_error = "unknown error"
    for wait_until, delay, timeout in _ATTEMPTS:
        try:
            result = await crawler.arun(url=url, config=_page_config(wait_until, delay, timeout))
        except Exception as e:  # timeouts etc.
            last_error = f"{type(e).__name__}: {e}"
            continue
        if not result.success:
            last_error = result.error_message or "crawl failed"
            continue
        text = clean_markdown(str(result.markdown or ""))
        if len(text) >= MIN_CHARS:
            return text
        last_error = f"page rendered only {len(text)} chars"
    raise RuntimeError(last_error)


# @tool
async def web_crawler(url: str) -> str:
    """
    Use this tool only when you want to scrape content live from the questohive site
    using https eg https://www.questohive.com/ or any other questohive related link
    """
    try:
        return await _fetch_page_text(url)
    except Exception as e:
        return f"Scrape error: {e}"


# ----------------------------------------------------------------------------
# Job list: capture the page's own API call, then replay it for every page
# ----------------------------------------------------------------------------
SKIP_HEADERS = {"host", "content-length", "connection", "accept-encoding"}
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
EXPECTED_RE = re.compile(r"(\d+)\s+roles found", re.I)


async def discover_api(url: str):
    crawler = await get_crawler()
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            wait_until="domcontentloaded",
            delay_before_return_html=8,
            page_timeout=60_000,
            capture_network_requests=True,
            verbose=False,
            excluded_tags=["script", "style", "nav", "footer", "svg"],
        ),
    )
    if not result.success:
        raise RuntimeError(result.error_message or "crawl failed")
    api_req = next(
        (
            r
            for r in (result.network_requests or [])
            if r.get("event_type") == "request"
            and API_HINT in r.get("url", "")
            and str(r.get("method", "")).upper() == "POST"
        ),
        None,
    )
    return result, api_req


def _call_api(api_req: dict, page: int, limit: int):
    parts = urlsplit(api_req["url"])
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["page"], query["limit"] = str(page), str(limit)
    url = urlunsplit(parts._replace(query=urlencode(query)))

    raw = api_req.get("post_data")
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                for key in ("page", "pageNumber", "page_number"):
                    if key in obj:
                        obj[key] = page
                for key in ("limit", "pageSize", "page_size", "per_page"):
                    if key in obj:
                        obj[key] = limit
            data = json.dumps(obj).encode()
        except ValueError:
            data = raw.encode()
    else:
        data = b"{}"

    headers = {
        k: v
        for k, v in (api_req.get("headers") or {}).items()
        if k.lower() not in SKIP_HEADERS and not k.startswith(":")
    }
    headers.setdefault("Content-Type", "application/json")
    headers.setdefault("Accept", "application/json")
    headers.setdefault("Origin", "https://www.micro1.ai")
    headers.setdefault("Referer", "https://www.micro1.ai/")
    headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0 Safari/537.36",
    )

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
    except (urllib.error.URLError, ValueError) as e:
        raise RuntimeError(str(e)) from e


def find_job_list(obj) -> list[dict]:
    """Largest list of dicts anywhere in the JSON = the jobs."""
    best: list[dict] = []

    def walk(o):
        nonlocal best
        if isinstance(o, list):
            if o and all(isinstance(x, dict) for x in o) and len(o) > len(best):
                best = o
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)

    walk(obj)
    return best


def find_total(obj) -> int | None:
    totals: list[int] = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, int) and not isinstance(v, bool) and re.search(r"total|count", k, re.I):
                    totals.append(v)
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(obj)
    return max(totals) if totals else None


def job_id(job: dict) -> str | None:
    for k in ("job_id", "id", "_id", "jobId", "uuid", "post_id", "postId"):
        v = job.get(k)
        if isinstance(v, str) and UUID_RE.match(v):
            return v
    return next((v for v in job.values() if isinstance(v, str) and UUID_RE.match(v)), None)


async def fetch_job_list(api_req: dict, expected: int | None) -> list[dict]:
    jobs: list[dict] = []
    seen: set[str] = set()

    def add(batch: list[dict]) -> int:
        new = 0
        for j in batch:
            key = job_id(j) or json.dumps(j, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                jobs.append(j)
                new += 1
        return new

    async def get(page: int, limit: int):
        return await asyncio.to_thread(_call_api, api_req, page, limit)

    page_size, total, batch = 18, expected, []
    for attempt in (100, 18):  # the API allows at most 100 per page
        try:
            data = await get(1, attempt)
        except RuntimeError as e:
            print(f"API request (limit={attempt}) failed: {e}")
            continue
        batch = find_job_list(data)
        total = find_total(data) or expected
        if batch:
            page_size = attempt
            if total and len(batch) < min(attempt, total):
                page_size = len(batch)  # server capped the page size
            break
    if not batch:
        raise RuntimeError("API returned no jobs")

    add(batch)
    target = total or expected or 10**9
    for page in range(2, 80):
        if len(jobs) >= target:
            break
        try:
            data = await get(page, page_size)
        except RuntimeError as e:
            print(f"Page {page} failed: {e}")
            break
        batch = find_job_list(data)
        if not batch or add(batch) == 0:
            break
        await asyncio.sleep(0.3)
    return jobs


# ----------------------------------------------------------------------------
# One clean record per job
# ----------------------------------------------------------------------------
_MAPPED = {
    "job_id", "job_name", "company_name", "date_posted", "ideal_hourly_rate", "skills",
    "role_type", "domain_slug", "job_type", "apply_url", "company_logo", "is_micro1_account",
}


def normalize(j: dict) -> dict:
    jid = job_id(j)
    rate = j.get("ideal_hourly_rate") if isinstance(j.get("ideal_hourly_rate"), dict) else {}
    return {
        "job_id": jid,
        "url": JOB_URL.format(jid) if jid else None,
        "title": j.get("job_name"),
        "company": j.get("company_name"),
        "posted": j.get("date_posted"),
        "pay_min": rate.get("min"),
        "pay_max": rate.get("max"),
        "pay_unit": "hour",
        "skills": j.get("skills") or [],
        "role_type": j.get("role_type"),
        "domain": j.get("domain_slug"),
        "job_type": j.get("job_type"),
        "apply_url": j.get("apply_url"),
        # any other API field we haven't mapped yet, so nothing is silently lost
        "extra": {k: v for k, v in j.items() if k not in _MAPPED and v not in (None, "", [], {})},
        # filled in by enrich_jobs()
        "description": None,
        "scrape_error": None,
    }


async def enrich_jobs(jobs: list[dict], concurrency: int = DEFAULT_CONCURRENCY) -> None:
    """Fetch each job page and store its cleaned content in job['description'] (in place)."""
    sem = asyncio.Semaphore(concurrency)
    done = 0
    total = len(jobs)

    async def one(job: dict) -> None:
        nonlocal done
        async with sem:
            try:
                job["description"] = await _fetch_page_text(job["url"])
            except Exception as e:
                job["scrape_error"] = str(e)
            done += 1
            status = "ok " if job["description"] else "ERR"
            print(f"  [{done}/{total}] {status} {job['title'] or job['job_id']}")

    await asyncio.gather(*(one(j) for j in jobs if j["url"]))


async def collect_jobs(limit: int | None = None, concurrency: int = DEFAULT_CONCURRENCY) -> list[dict]:
    """Main entry point: returns the full records (metadata + description) for every job."""
    try:
        result, api_req = await discover_api(LIST_URL)
        if not api_req:
            raise RuntimeError("Could not capture the job-list API call from the page")
        em = EXPECTED_RE.search(str(result.markdown or ""))
        expected = int(em.group(1)) if em else None

        raw = await fetch_job_list(api_req, expected)
        print(f"Job list: {len(raw)} jobs" + (f" (site says {expected})" if expected else ""))
        jobs = [normalize(j) for j in raw]
        if limit:
            jobs = jobs[:limit]
            print(f"Limiting to the first {len(jobs)} jobs")

        print(f"Scraping job pages (concurrency={concurrency}) ...")
        await enrich_jobs(jobs, concurrency)
        return jobs
    finally:
        await close_crawler()


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def _pay(job: dict) -> str:
    lo, hi = job["pay_min"], job["pay_max"]
    if lo is None:
        return "n/a"
    return f"${lo:g}/h" if lo == hi or hi is None else f"${lo:g}-{hi:g}/h"


def print_jobs(jobs: list[dict], full: bool = False) -> None:
    print("\n" + "=" * 80)
    for i, j in enumerate(jobs, 1):
        print(f"[{i}] {j['title']}  ({j['company']})")
        print(f"    url     : {j['url']}")
        print(f"    pay     : {_pay(j)}")
        print(f"    posted  : {j['posted']}")
        print(f"    type    : {j['job_type']} / {j['domain']} / {j['role_type']}")
        print(f"    skills  : {', '.join(j['skills'])}")
        if j["extra"]:
            print(f"    extra   : {json.dumps(j['extra'], ensure_ascii=False)[:300]}")
        if j["description"]:
            text = j["description"]
            shown = text if full or len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS].rstrip() + " …"
            print(f"    content : ({len(text)} chars)")
            print("      " + shown.replace("\n", "\n      "))
        else:
            print(f"    content : NOT SCRAPED -> {j['scrape_error']}")
        print("-" * 80)

    ok = sum(1 for j in jobs if j["description"])
    failed = [j for j in jobs if not j["description"]]
    print(f"Total: {len(jobs)} jobs | content scraped: {ok} | failed: {len(failed)}")
    for j in failed:
        print(f"  FAILED {j['url']}  ({j['scrape_error']})")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process the first N jobs")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--full", action="store_true", help="print full descriptions")
    args = ap.parse_args()

    jobs = await collect_jobs(limit=args.limit, concurrency=args.concurrency)
    print_jobs(jobs, full=args.full)


if __name__ == "__main__":
    asyncio.run(main())