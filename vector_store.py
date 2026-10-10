"""
Qdrant + FastEmbed layer for job postings.

Layout (same idea as your pastQuestions store):
  * Lazy singletons  : importing this module loads nothing heavy. The Qdrant client, embedding
                       model and reranker are created on first use and then reused.
  * sync_jobs()      : WRITE path. Makes Qdrant match a list of scraped jobs (idempotent).
  * search_jobs()    : READ path. Natural-language query in, ranked job dicts out.
                       (job_tools.py wraps this into the string-in/string-out agent tools.)

Retrieval design
  * ONE JOB = ONE POINT. The full posting lives in the payload, never chunked.
  * The vector is built from a compact "search card": a header (title, field, level, pay,
    skills) + the description with its duplicated boilerplate removed, so the limited
    embedding window is spent on what the job actually IS.
  * Search = vector search for ~20 candidates -> cross-encoder reranker -> top N.
"""
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

load_dotenv()
logger = logging.getLogger(__name__)

COLLECTION = os.getenv("QDRANT_COLLECTION", "micro1-jobs")
SOURCE = "micro1_jobs"  # tag on every point we write, so we never touch other data in a shared collection

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Cross-encoder that re-scores the top candidates. Set RERANK_MODEL= (empty) to disable.
RERANK_MODEL = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v1-turbo-en")
CACHE_DIR = Path(os.getenv("FASTEMBED_CACHE", Path.home() / ".cache" / "fastembed"))

# Bump when the embedded text recipe changes: it's part of the content hash, so the next sync
# automatically re-embeds everything instead of reporting "unchanged".
EMBED_RECIPE = "2"

_LONG_WINDOW = ("jina", "nomic", "arctic-embed-m-long")
MAX_EMBED_CHARS = int(
    os.getenv("MAX_EMBED_CHARS", 24_000 if any(k in EMBED_MODEL.lower() for k in _LONG_WINDOW) else 1_800)
)
RERANK_CHARS = int(os.getenv("RERANK_CHARS", 1_500))

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

PAYLOAD_INDEXES = {
    "source": models.PayloadSchemaType.KEYWORD,
    "domain": models.PayloadSchemaType.KEYWORD,
    "role_type": models.PayloadSchemaType.KEYWORD,
    "job_type": models.PayloadSchemaType.KEYWORD,
    "skills_norm": models.PayloadSchemaType.KEYWORD,
    "pay_min": models.PayloadSchemaType.FLOAT,
    "pay_max": models.PayloadSchemaType.FLOAT,
    "posted_at": models.PayloadSchemaType.DATETIME,
}

_SOURCE_FILTER = models.Filter(
    must=[models.FieldCondition(key="source", match=models.MatchValue(value=SOURCE))]
)


# ----------------------------------------------------------------------------
# Lazy singletons
# ----------------------------------------------------------------------------
class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL, model=None):
        self.model_name = model_name
        self._model = model
        self._dim: int | None = None
        n = model_name.lower()
        # fastembed does NOT add BGE's retrieval instruction for queries, so we do.
        self._query_prefix = _BGE_QUERY_PREFIX if n.startswith("baai/bge-") and "-en" in n else ""

    @property
    def model(self):
        if self._model is None:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Loading embedding model %s (first run downloads it) ...", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(CACHE_DIR))
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(next(iter(self.model.embed(["dimension probe"]))))
        return self._dim

    def embed_documents(self, texts: list[str], batch_size: int = 4) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts, batch_size=batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(self._query_prefix + text))).tolist()


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL, model=None):
        self.model_name = model_name
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Loading reranker %s (first run downloads it) ...", self.model_name)
            self._model = TextCrossEncoder(model_name=self.model_name, cache_dir=str(CACHE_DIR))
        return self._model

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, docs)]


_client: QdrantClient | None = None
_embedder: Embedder | None = None
_reranker: Reranker | None = None


def get_client() -> QdrantClient:
    """Qdrant Cloud / server if QDRANT_URL is set, otherwise an on-disk local database."""
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL")
        if url:
            _client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=60)
        else:
            path = os.getenv("QDRANT_PATH", "qdrant_data")
            logger.warning("QDRANT_URL not set -> using local on-disk Qdrant in ./%s", path)
            _client = QdrantClient(path=path)
    return _client


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_reranker() -> Reranker | None:
    global _reranker
    if _reranker is None and RERANK_MODEL:
        _reranker = Reranker()
    return _reranker


# ----------------------------------------------------------------------------
# Building what gets embedded / stored
# ----------------------------------------------------------------------------
def _iso(ts) -> str | None:
    """'2026-08-10 12:28:30' -> '2026-08-10T12:28:30Z' (Qdrant datetime format)."""
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


def pay_text(d: dict) -> str | None:
    lo, hi = d.get("pay_min"), d.get("pay_max")
    if lo is None:
        return None
    return f"${lo:g}/hour" if hi is None or lo == hi else f"${lo:g}-{hi:g}/hour"


def _clean_skills(skills) -> list[str]:
    out, seen = [], set()
    for s in skills or []:
        s = re.sub(r"\s+", " ", str(s)).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


_PAY_LINE = re.compile(r"^\$[\d,.]+(\s*-\s*\$?[\d,.]+)?\s*/?\s*\w*$")


def clean_description(desc: str | None, d: dict) -> str:
    """
    For EMBEDDING only (the payload keeps the original). Job pages open by repeating the title,
    the pay line, a 'Required Skills' list and a 'Role Title' line, all of which we already put
    in the header. Dropping them spends the small embedding window on real content.
    """
    if not desc:
        return ""
    title = (d.get("title") or "").strip().lower()
    skills = {s.lower() for s in _clean_skills(d.get("skills"))}
    out, in_skills = [], False
    for line in desc.splitlines():
        s = line.strip()
        low = s.lower()
        if not s:
            out.append("")
            continue
        if title and re.sub(r"^#+\s*", "", low).strip() == title:
            continue
        if low.startswith("$") and _PAY_LINE.match(re.sub(r"pay$", "", low).strip()):
            continue
        if re.match(r"^#{1,6}\s*required skills\s*$", low):
            in_skills = True
            continue
        if in_skills:
            if low in skills:
                continue
            in_skills = False
        if low.startswith("**role title:**"):
            continue
        out.append(s)
    text = "\n".join(out)
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _header(d: dict) -> str:
    skills = _clean_skills(d.get("skills"))
    domain = (d.get("domain") or "").replace("-", " ")
    lines = [
        f"Title: {d['title']}" if d.get("title") else None,
        f"Company: {d['company']}" if d.get("company") else None,
        f"Field: {domain}" if domain else None,
        f"Level: {d['role_type']}" if d.get("role_type") else None,
        f"Pay: {pay_text(d)}" if pay_text(d) else None,
        f"Skills: {', '.join(skills)}" if skills else None,
    ]
    return "\n".join(l for l in lines if l)


def build_embedding_text(job: dict) -> tuple[str, bool]:
    text = f"{_header(job)}\n\n{clean_description(job.get('description'), job)}".strip()
    return (text[:MAX_EMBED_CHARS], True) if len(text) > MAX_EMBED_CHARS else (text, False)


def build_rerank_text(payload: dict) -> str:
    return f"{_header(payload)}\n\n{clean_description(payload.get('description'), payload)[:RERANK_CHARS]}".strip()


def to_payload(job: dict) -> dict:
    skills = _clean_skills(job.get("skills"))
    payload = {
        "source": SOURCE,
        "job_id": job.get("job_id"),
        "url": job.get("url"),
        "apply_url": job.get("apply_url"),
        "title": job.get("title"),
        "company": job.get("company"),
        "posted_at": _iso(job.get("posted")),
        "pay_min": job.get("pay_min"),
        "pay_max": job.get("pay_max"),
        "pay_unit": job.get("pay_unit") or "hour",
        "skills": skills,
        "skills_norm": [s.lower() for s in skills],
        "domain": job.get("domain"),
        "role_type": job.get("role_type"),
        "job_type": job.get("job_type"),
        "extra": job.get("extra") or {},
        "has_description": bool(job.get("description")),
        "description": job.get("description"),  # FULL text, never chunked
    }
    payload["content_hash"] = hashlib.sha256(
        json.dumps([payload, EMBED_MODEL, EMBED_RECIPE], sort_keys=True, default=str).encode()
    ).hexdigest()
    return payload


# ----------------------------------------------------------------------------
# Collection management
# ----------------------------------------------------------------------------
def ensure_collection(recreate: bool = False) -> None:
    client, dim = get_client(), get_embedder().dim
    exists = client.collection_exists(COLLECTION)
    if exists and recreate:
        client.delete_collection(COLLECTION)
        exists = False

    if exists:
        size = getattr(client.get_collection(COLLECTION).config.params.vectors, "size", None)
        if size is not None and size != dim:
            raise RuntimeError(
                f"Collection '{COLLECTION}' has {size}-dim vectors but {EMBED_MODEL} produces {dim}. "
                "You changed the embedding model: use a new QDRANT_COLLECTION name or run with --recreate."
            )
    else:
        client.create_collection(
            COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

    for field, schema in PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)
        except Exception:
            pass  # already exists


def existing_hashes() -> dict[str, str | None]:
    """{point_id: content_hash} for the points THIS pipeline wrote."""
    client = get_client()
    if not client.collection_exists(COLLECTION):
        return {}
    out: dict[str, str | None] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION, scroll_filter=_SOURCE_FILTER, limit=256, offset=offset,
            with_payload=["content_hash"], with_vectors=False,
        )
        for p in points:
            out[str(p.id)] = (p.payload or {}).get("content_hash")
        if offset is None:
            return out


def count_points() -> tuple[int, int]:
    """(all points in the collection, points written by this pipeline)."""
    client = get_client()
    if not client.collection_exists(COLLECTION):
        return 0, 0
    total = client.count(COLLECTION).count
    ours = client.count(COLLECTION, count_filter=_SOURCE_FILTER).count
    return total, ours


# ----------------------------------------------------------------------------
# WRITE path
# ----------------------------------------------------------------------------
def sync_jobs(
    jobs: list[dict],
    force: bool = False,
    prune: bool = False,
    recreate: bool = False,
    batch_size: int = 16,
    max_prune_fraction: float = 0.3,
) -> dict:
    """Make Qdrant match `jobs`: upsert new/changed, optionally delete the ones no longer listed."""
    ensure_collection(recreate=recreate)
    embedder, client = get_embedder(), get_client()
    have = existing_hashes()

    todo, unchanged, invalid, trimmed = [], 0, 0, 0
    for job in jobs:
        jid = job.get("job_id")
        if not jid or not UUID_RE.match(jid):
            invalid += 1
            continue
        payload = to_payload(job)
        if not force and have.get(jid) == payload["content_hash"]:
            unchanged += 1
            continue
        text, was_trimmed = build_embedding_text(job)
        trimmed += was_trimmed
        todo.append((jid, payload, text))

    for i in range(0, len(todo), batch_size):
        chunk = todo[i : i + batch_size]
        vectors = embedder.embed_documents([t[2] for t in chunk])
        client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(id=jid, vector=vec, payload=payload)
                for (jid, payload, _), vec in zip(chunk, vectors)
            ],
            wait=True,
        )
        print(f"  embedded + stored {min(i + batch_size, len(todo))}/{len(todo)}")

    pruned = 0
    if prune:
        keep = {j["job_id"].lower() for j in jobs if j.get("job_id")}
        stale = [i for i in have if i.lower() not in keep]
        if stale:
            if len(stale) > max(5, int(max_prune_fraction * len(have))):
                raise RuntimeError(
                    f"Refusing to delete {len(stale)} of {len(have)} jobs: the scrape may be incomplete."
                )
            client.delete(COLLECTION, points_selector=models.PointIdsList(points=stale), wait=True)
            pruned = len(stale)

    return {"upserted": len(todo), "unchanged": unchanged, "invalid": invalid, "trimmed": trimmed, "pruned": pruned}


# ----------------------------------------------------------------------------
# READ path
# ----------------------------------------------------------------------------
def _summary(payload: dict, n: int = 350) -> str:
    text = re.sub(r"\s+", " ", clean_description(payload.get("description"), payload)).strip()
    if len(text) <= n:
        return text
    cut = text[:n]
    return cut[: cut.rfind(" ")] + "…" if " " in cut else cut + "…"


def _hit(payload: dict, score: float, vector_score: float) -> dict:
    return {
        "job_id": payload.get("job_id"),
        "title": payload.get("title"),
        "url": payload.get("url"),
        "apply_url": payload.get("apply_url"),
        "pay": pay_text(payload),
        "pay_min": payload.get("pay_min"),
        "pay_max": payload.get("pay_max"),
        "skills": payload.get("skills") or [],
        "domain": payload.get("domain"),
        "role_type": payload.get("role_type"),
        "posted_at": payload.get("posted_at"),
        "summary": _summary(payload),
        "score": round(score, 4),
        "vector_score": round(vector_score, 4),
    }


def search_jobs(
    query: str,
    limit: int = 5,
    *,
    min_pay: float | None = None,   # keep jobs whose TOP pay reaches this ("pays at least X")
    max_pay: float | None = None,   # keep jobs whose BASE pay is at most this
    domain: str | list[str] | None = None,
    skills: list[str] | None = None,
    posted_after: datetime | str | None = None,
    rerank: bool | None = None,     # None = use the configured reranker if there is one
    candidates: int = 20,
) -> list[dict]:
    must = [models.FieldCondition(key="source", match=models.MatchValue(value=SOURCE))]
    if domain:
        values = [domain] if isinstance(domain, str) else list(domain)
        must.append(models.FieldCondition(key="domain", match=models.MatchAny(any=values)))
    if min_pay is not None:
        must.append(models.FieldCondition(key="pay_max", range=models.Range(gte=float(min_pay))))
    if max_pay is not None:
        must.append(models.FieldCondition(key="pay_min", range=models.Range(lte=float(max_pay))))
    if skills:
        must.append(models.FieldCondition(key="skills_norm", match=models.MatchAny(any=[s.lower() for s in skills])))
    if posted_after:
        must.append(models.FieldCondition(key="posted_at", range=models.DatetimeRange(gte=posted_after)))

    client = get_client()
    if not client.collection_exists(COLLECTION):
        raise RuntimeError(f"Collection '{COLLECTION}' doesn't exist yet. Run `python ingest.py` first.")

    reranker = get_reranker() if rerank in (None, True) else None
    pool = max(limit, candidates) if reranker else limit
    points = client.query_points(
        COLLECTION,
        query=get_embedder().embed_query(query),
        query_filter=models.Filter(must=must),
        limit=pool,
        with_payload=True,
    ).points
    if not points:
        return []

    if reranker and len(points) > 1:
        scores = reranker.score(query, [build_rerank_text(p.payload or {}) for p in points])
        ranked = sorted(zip(points, scores), key=lambda x: x[1], reverse=True)
        return [_hit(p.payload or {}, s, p.score) for p, s in ranked[:limit]]
    return [_hit(p.payload or {}, p.score, p.score) for p in points[:limit]]


def get_job(job_id: str) -> dict | None:
    """Full stored record (including the complete description) for one job id."""
    if not job_id or not UUID_RE.match(job_id.strip()):
        return None
    got = get_client().retrieve(COLLECTION, ids=[job_id.strip().lower()], with_payload=True)
    if not got or (got[0].payload or {}).get("source") != SOURCE:
        return None
    return got[0].payload