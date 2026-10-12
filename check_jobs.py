"""
Health check for scraped_jobs.json: how many jobs have a readable description?

    python check_jobs.py

If many jobs come out as "header only", the problem is in web_scraper.py (it captured the page
header but not the description text), not in the website.
"""
import json

import jobs_data
from classify import enrich


def ascii_safe(text: str) -> str:
    return text.encode("ascii", "replace").decode()


def main() -> None:
    raw = json.loads(jobs_data.JOBS_FILE.read_text(encoding="utf-8"))
    no_text, header_only, short, errors, fine = [], [], [], [], []
    for item in raw:
        job = enrich(item)
        desc = item.get("description") or ""
        if item.get("scrape_error"):
            errors.append((item, len(desc)))
        if not desc:
            no_text.append((item, 0))
        elif not job["has_description"]:
            header_only.append((item, len(desc)))
        elif len(job["body"]) < 400:
            short.append((item, len(job["body"])))
        else:
            fine.append(item)

    print(f"File: {jobs_data.JOBS_FILE}")
    print(f"Jobs in file:                       {len(raw)}")
    print(f"  readable description:             {len(fine)}")
    print(f"  short description (<400 chars):   {len(short)}")
    print(f"  HEADER ONLY (no real text):       {len(header_only)}")
    print(f"  no description at all:            {len(no_text)}")
    print(f"  scrape_error set:                 {len(errors)}")

    for label, rows in (("HEADER ONLY", header_only), ("NO DESCRIPTION", no_text), ("SHORT", short)):
        if rows:
            print(f"\n{label} (first 8):")
            for item, n in rows[:8]:
                print(f"  {item['job_id'][:8]}  {n:>5} chars  {ascii_safe(item.get('title') or '')}  err={item.get('scrape_error')}")

    if header_only or no_text:
        print("\nTo re-scrape only these jobs (patiently, 2 pages at a time) and update the website and search:")
        print("    python ingest.py --retry-missing")

    if header_only:
        item = header_only[0][0]
        print(f"\nWhat was scraped for '{ascii_safe(item['title'])}':")
        print(ascii_safe((item.get("description") or "")[:500]))


if __name__ == "__main__":
    main()