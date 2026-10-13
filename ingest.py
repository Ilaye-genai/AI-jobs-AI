"""Collect Micro1 + Mercor jobs, save one JSON snapshot, and sync them to Qdrant.

Examples:
    python ingest.py
    python ingest.py --limit 5
    python ingest.py --from-snapshot
    python ingest.py --retry-missing
    python ingest.py --force
    python ingest.py --prune
    python ingest.py --recreate
    python ingest.py --search "remote AI evaluation job" --top 5

The default collection run scrapes both platforms. Both sets are stored together in
scraped_jobs.json and passed together to vector_store.sync_jobs().
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import vector_store as vs

SNAPSHOT = Path("scraped_jobs.json")
DEFAULT_CONCURRENCY = 4

SOURCE_MODULES = (
    ("micro1", "web_scraper"),
    ("mercor", "mercor"),
)


def _save_snapshot(jobs: list[dict]) -> None:
    """Atomically write the combined snapshot so readers never see half a file."""
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = SNAPSHOT.with_name(SNAPSHOT.name + ".tmp")
    temporary.write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, SNAPSHOT)


def _infer_source(job: dict) -> str:
    source = str(job.get("source") or "").strip().lower()
    if source in {"micro1", "mercor"}:
        return source

    url = str(job.get("url") or job.get("apply_url") or "").lower()
    if "mercor.com" in url:
        return "mercor"
    if "micro1.ai" in url:
        return "micro1"

    # Old snapshots created before source tagging were Micro1-only.
    return "micro1"


def _is_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _canonicalise_job(job: dict) -> dict:
    """Add source metadata and a stable UUID where Qdrant needs one.

    Micro1 already supplies UUID job IDs, so those IDs are preserved to keep existing
    Qdrant points stable. Mercor IDs such as 'list_AAAB...' are retained as
    source_job_id and mapped to a deterministic UUID for vector-store compatibility.
    """
    result = dict(job)
    source = _infer_source(result)
    native_id = result.get("source_job_id") or result.get("job_id")
    if native_id is None:
        native_id = result.get("url") or "|".join(
            str(result.get(key) or "") for key in ("title", "company", "posted")
        )
    native_id = str(native_id).strip()

    result["source"] = source
    result["source_job_id"] = native_id or None

    if source == "micro1" and _is_uuid(str(result.get("job_id") or "")):
        result["job_id"] = str(uuid.UUID(str(result["job_id"])))
    else:
        stable_key = f"job-collector:{source}:{native_id}"
        result["job_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))

    result.setdefault("skills", [])
    if not isinstance(result["skills"], list):
        value = result["skills"]
        result["skills"] = [str(value)] if value not in (None, "") else []

    result.setdefault("extra", {})
    if not isinstance(result["extra"], dict):
        result["extra"] = {}

    result.setdefault("description", None)
    result.setdefault("scrape_error", None)
    return result


def _record_key(job: dict) -> tuple[str, str]:
    source = _infer_source(job)
    source_id = job.get("source_job_id") or job.get("job_id") or job.get("url")
    return source, str(source_id)


def _prefer_old_description(fresh_jobs: list[dict]) -> list[dict]:
    """Keep a previously good description if a refresh fails for that same listing."""
    if not SNAPSHOT.exists():
        return fresh_jobs

    try:
        old_data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fresh_jobs
    if not isinstance(old_data, list):
        return fresh_jobs

    previous: dict[tuple[str, str], dict] = {}
    for old in old_data:
        if isinstance(old, dict):
            prepared = _canonicalise_job(old)
            previous[_record_key(prepared)] = prepared

    merged: list[dict] = []
    for fresh in fresh_jobs:
        old = previous.get(_record_key(fresh))
        current = dict(fresh)
        if old and not current.get("description") and old.get("description"):
            current["description"] = old["description"]
            # If fresh scrape failed but the previous body is useful, preserve the
            # previous scrape state rather than marking the good body as a failure.
            current["scrape_error"] = None
        merged.append(current)
    return merged


def _deduplicate(jobs: list[dict]) -> list[dict]:
    """Deduplicate records by platform + native listing ID while preserving order."""
    by_key: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []

    for original in jobs:
        if not isinstance(original, dict):
            continue
        job = _canonicalise_job(original)
        key = _record_key(job)
        if key not in by_key:
            by_key[key] = job
            order.append(key)
            continue

        current = by_key[key]
        # Keep the best available text if duplicate API records occur.
        if len(str(job.get("description") or "")) > len(str(current.get("description") or "")):
            current["description"] = job.get("description")
            current["scrape_error"] = job.get("scrape_error")
        # Refresh metadata with the latest record, but don't erase useful old values
        # with None or empty strings.
        for field, value in job.items():
            if value not in (None, "", [], {}) or field not in current:
                if field not in {"description", "scrape_error"}:
                    current[field] = value

    return [by_key[key] for key in order]


def scrape_all(limit: int | None, concurrency: int) -> list[dict]:
    """Scrape both job sources and return one combined, normalized list."""
    combined: list[dict] = []

    for source, module_name in SOURCE_MODULES:
        module = importlib.import_module(module_name)
        collector = getattr(module, "collect_jobs", None)
        if collector is None:
            raise RuntimeError(f"{module_name}.py does not expose collect_jobs().")

        print(f"\n{'=' * 24} Scraping {source.upper()} {'=' * 24}")
        # Each scraper owns and closes its browser. Run them sequentially so the two
        # independent Playwright/Crawl4AI browser lifecycles don't overlap on Windows.
        jobs = asyncio.run(collector(limit=limit, concurrency=concurrency))
        for job in jobs:
            if not isinstance(job, dict):
                continue
            tagged = dict(job)
            tagged["source"] = source
            tagged.setdefault("source_job_id", tagged.get("job_id"))
            combined.append(tagged)
        print(f"Collected {len(jobs)} {source} jobs.")

    combined = _deduplicate(combined)
    combined = _prefer_old_description(combined)

    # A limited run is a test/partial refresh, not a full catalogue replacement.
    # Preserve records outside the sample so --limit cannot accidentally erase the
    # rest of an existing shared snapshot. A full run refreshes the complete catalogue.
    if limit is not None and SNAPSHOT.exists():
        try:
            previous_data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_data = []
        if isinstance(previous_data, list):
            before_count = len(combined)
            combined = _deduplicate(previous_data + combined)
            print(
                f"Partial scrape merged with the existing snapshot: "
                f"{before_count} fresh jobs; {len(combined)} total records retained."
            )

    _save_snapshot(combined)
    print(f"\nSaved {len(combined)} combined jobs to {SNAPSHOT}.")
    return combined


def _load_snapshot() -> list[dict]:
    if not SNAPSHOT.exists():
        raise SystemExit(f"{SNAPSHOT} does not exist. Run `python ingest.py` first.")

    try:
        data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse {SNAPSHOT}: {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit(f"{SNAPSHOT} must contain a JSON list of job records.")

    jobs = _deduplicate(data)
    print(f"Loaded {len(jobs)} jobs from {SNAPSHOT}")
    # Rewrite older snapshots once so source tags and stable IDs are persisted.
    _save_snapshot(jobs)
    return jobs


def retry_missing(jobs: list[dict], concurrency: int | None) -> None:
    """Retry missing descriptions through the correct scraper for each source."""
    concurrency = concurrency or DEFAULT_CONCURRENCY
    fixed_total = 0
    remaining_total = 0

    for source, module_name in SOURCE_MODULES:
        source_jobs = [job for job in jobs if _infer_source(job) == source]
        if not source_jobs:
            continue

        module = importlib.import_module(module_name)
        needs_retry = getattr(module, "needs_retry", None)
        retry_fn = getattr(module, "retry_missing", None)
        if not callable(needs_retry) or not callable(retry_fn):
            print(f"Skipping retry for {source}: {module_name}.py has no retry support.")
            continue

        missing = [job for job in source_jobs if needs_retry(job)]
        if not missing:
            print(f"{source}: every job already has a usable description.")
            continue

        source_concurrency = min(concurrency, getattr(module, "RETRY_CONCURRENCY", 2))
        print(f"\nRetrying {len(missing)} {source} jobs (concurrency={source_concurrency}) ...")
        fixed = asyncio.run(retry_fn(missing, source_concurrency))
        fixed_total += int(fixed or 0)
        remaining_total += sum(1 for job in missing if needs_retry(job))

    _save_snapshot(jobs)
    print(f"Saved retry results to {SNAPSHOT}. Fixed {fixed_total}; still missing descriptions: {remaining_total}.")


def run_ingest(args: argparse.Namespace) -> None:
    if args.prune and args.limit is not None:
        raise SystemExit("--prune cannot be combined with --limit; it could delete other existing jobs.")
    if args.retry_missing and args.limit is not None:
        raise SystemExit("--retry-missing cannot be combined with --limit; it works on the whole snapshot.")
    if args.recreate and args.limit is not None:
        raise SystemExit("--recreate cannot be combined with --limit; rebuild only from a complete dataset.")
    if args.from_snapshot and not SNAPSHOT.exists():
        raise SystemExit(f"{SNAPSHOT} does not exist. Run `python ingest.py` first.")

    if args.from_snapshot or args.retry_missing:
        jobs = _load_snapshot()
    else:
        jobs = scrape_all(args.limit, args.concurrency or DEFAULT_CONCURRENCY)

    if args.retry_missing:
        retry_missing(jobs, args.concurrency)

    jobs = _deduplicate(jobs)
    # Ensure the exact combined dataset being embedded is also the dataset on disk.
    _save_snapshot(jobs)

    no_text = sum(1 for job in jobs if not job.get("description"))
    if no_text:
        print(
            f"Note: {no_text} jobs have no usable description; they will still be stored "
            "with metadata and can be retried later with --retry-missing."
        )

    print(f"\nEmbedding {len(jobs)} Micro1 + Mercor jobs with {vs.EMBED_MODEL} into '{vs.COLLECTION}' ...")
    stats = vs.sync_jobs(
        jobs,
        force=args.force,
        prune=args.prune,
        recreate=args.recreate,
    )
    stats = stats or {}
    print(
        f"Stored {stats.get('upserted', 0)} | unchanged {stats.get('unchanged', 0)} | "
        f"invalid id {stats.get('invalid', 0)} | pruned {stats.get('pruned', 0)} | "
        f"long descriptions trimmed for vectors {stats.get('trimmed', 0)} "
        "(the full description should remain in each Qdrant payload)."
    )

    total, ours = vs.count_points()
    print(f"Collection '{vs.COLLECTION}': {total} points in total; {ours} job records from this pipeline.")
    if total > ours:
        print(
            f"  {total - ours} other points are outside this job pipeline.\n"
            "  Keep job postings in their own Qdrant collection if you need strict isolation."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Micro1 and Mercor job postings into Qdrant.")
    parser.add_argument("--limit", type=int, default=None, help="scrape the first N jobs from each site")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="browser tabs per scraper")
    parser.add_argument("--from-snapshot", action="store_true", help="skip scraping and reuse scraped_jobs.json")
    parser.add_argument("--retry-missing", action="store_true", help="retry records with no usable description")
    parser.add_argument("--force", action="store_true", help="re-embed every record even if unchanged")
    parser.add_argument("--prune", action="store_true", help="delete previously indexed pipeline jobs not in this scrape")
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the configured collection")
    parser.add_argument("--search", metavar="QUERY", help="search existing indexed jobs instead of ingesting")
    parser.add_argument("--top", type=int, default=5, help="number of search results")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be zero or greater")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.top < 1:
        parser.error("--top must be at least 1")

    if args.search:
        from job_tools import find_jobs
        print(find_jobs(args.search, limit=args.top))
    else:
        run_ingest(args)


if __name__ == "__main__":
    main()
