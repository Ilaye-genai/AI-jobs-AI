"""
Scrape micro1 jobs -> embed with FastEmbed -> sync into Qdrant.   (your "sync_job.py")

    python ingest.py                     # scrape everything, embed, store
    python ingest.py --limit 5           # quick test with 5 jobs
    python ingest.py --from-snapshot     # skip scraping, reuse scraped_jobs.json (use this after changing the embedding recipe)
    python ingest.py --force             # re-embed everything even if unchanged
    python ingest.py --prune             # also delete jobs that are no longer listed
    python ingest.py --recreate          # drop + rebuild the collection

    python ingest.py --search "remote medical coding job paying over $50/hour"   # exactly what the agent tool returns
"""
import argparse
import asyncio
import json
from pathlib import Path

import vector_store as vs

SNAPSHOT = Path("scraped_jobs.json")


def scrape(limit: int | None, concurrency: int) -> list[dict]:
    import web_scraper as w  # imported here so --search works without the browser stack

    jobs = asyncio.run(w.collect_jobs(limit=limit, concurrency=concurrency))
    SNAPSHOT.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved a copy of the scrape to {SNAPSHOT} (reuse it with --from-snapshot)")
    return jobs


def run_ingest(args) -> None:
    if args.prune and args.limit:
        raise SystemExit("--prune can't be combined with --limit (it would delete everything else).")

    if args.from_snapshot:
        jobs = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        print(f"Loaded {len(jobs)} jobs from {SNAPSHOT}")
    else:
        jobs = scrape(args.limit, args.concurrency)

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
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--from-snapshot", action="store_true")
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