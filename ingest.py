"""
Scrape micro1 jobs -> embed with FastEmbed -> sync into Qdrant.   (your "sync_job.py")

    python ingest.py                     # scrape everything, embed, store
    python ingest.py --limit 5           # quick test with 5 jobs
    python ingest.py --from-snapshot     # skip scraping, reuse scraped_jobs.json (use this after changing the embedding recipe)
    python ingest.py --retry-missing     # re-scrape ONLY the jobs with no description (see check_jobs.py), then store
    python ingest.py --force             # re-embed everything even if unchanged
    python ingest.py --prune             # also delete jobs that are no longer listed
    python ingest.py --recreate          # drop + rebuild the collection

    python ingest.py --search "remote medical coding job paying over $50/hour"   # exactly what the agent tool returns
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

import vector_store as vs

SNAPSHOT = Path("scraped_jobs.json")
DEFAULT_CONCURRENCY = 4


def _save_snapshot(jobs: list[dict]) -> None:
    """Write scraped_jobs.json in one step. The website reloads this file whenever it changes, so it
    must never be able to see a half-written one: write a temp file, then swap it into place."""
    tmp = SNAPSHOT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, SNAPSHOT)


def scrape(limit: int | None, concurrency: int) -> list[dict]:
    import web_scraper as w  # imported here so --search works without the browser stack

    jobs = asyncio.run(w.collect_jobs(limit=limit, concurrency=concurrency))
    _save_snapshot(jobs)
    print(f"Saved a copy of the scrape to {SNAPSHOT} (reuse it with --from-snapshot)")
    return jobs


def retry_missing(jobs: list[dict], concurrency: int | None) -> None:
    """Re-scrape only the jobs that have no real description, update `jobs` in place and save the snapshot."""
    import web_scraper as w

    missing = sum(1 for j in jobs if w.needs_retry(j))
    if not missing:
        print("Every job already has a description. Nothing to retry.")
        return
    fixed = asyncio.run(w.retry_missing(jobs, concurrency or w.RETRY_CONCURRENCY))
    _save_snapshot(jobs)
    print(f"Saved the updated scrape to {SNAPSHOT}.")
    if fixed < missing:
        print(f"{missing - fixed} jobs still have no description. Run `python check_jobs.py` to see what their pages said.")


def run_ingest(args) -> None:
    if args.prune and args.limit:
        raise SystemExit("--prune can't be combined with --limit (it would delete everything else).")
    if args.retry_missing and args.limit:
        raise SystemExit("--retry-missing can't be combined with --limit: it works on the whole of scraped_jobs.json.")

    if args.from_snapshot or args.retry_missing:
        if not SNAPSHOT.exists():
            raise SystemExit(f"{SNAPSHOT} doesn't exist yet. Run `python ingest.py` first.")
        jobs = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        print(f"Loaded {len(jobs)} jobs from {SNAPSHOT}")
    else:
        jobs = scrape(args.limit, args.concurrency or DEFAULT_CONCURRENCY)

    if args.retry_missing:
        retry_missing(jobs, args.concurrency)

    no_text = sum(1 for j in jobs if not j.get("description"))
    if no_text:
        print(f"Note: {no_text} jobs have no scraped description; stored with metadata only, "
              "completed automatically on a later run.")

    print(f"Embedding with {vs.EMBED_MODEL} into collection '{vs.COLLECTION}' ...")
    stats = vs.sync_jobs(jobs, force=args.force, prune=args.prune, recreate=args.recreate)
    print(f"Stored {stats['upserted']} | unchanged {stats['unchanged']} | invalid id {stats['invalid']} | "
          f"pruned {stats['pruned']} | long descriptions trimmed for the vector {stats['trimmed']} "
          "(full text always kept in the payload)")

    total, ours = vs.count_points()
    print(f"Collection '{vs.COLLECTION}': {total} points in total, {ours} are job postings from this pipeline.")
    if total > ours:
        print(f"  {total - ours} other points are not from this pipeline. They are never searched or modified here.\n"
              "  Tip: keep jobs in their own collection (QDRANT_COLLECTION) to avoid mixing data types.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, help=f"pages scraped at once (default {DEFAULT_CONCURRENCY}; 2 with --retry-missing)")
    ap.add_argument("--from-snapshot", action="store_true")
    ap.add_argument("--retry-missing", action="store_true", help="re-scrape only the jobs that have no description")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--search", metavar="QUERY")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    if args.search:
        from job_tools import find_jobs

        print(find_jobs(args.search, limit=args.top))
    else:
        run_ingest(args)


if __name__ == "__main__":
    main()