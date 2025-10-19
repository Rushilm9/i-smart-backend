# app/routers/research.py
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import asyncio
import json
import math
import re
import random
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from core.database import get_db
# If available, import SessionLocal for background DB work
try:
    from core.database import SessionLocal  # type: ignore
except Exception:
    SessionLocal = None  # background authors will be disabled if not present

from model.models import Project, Paper, Author, PaperAuthor, Recommendation

# ── Rich console setup ─────────────────────────────────────────────────────────
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.traceback import install as rich_traceback
from rich.pretty import Pretty

rich_traceback(show_locals=False)
console = Console(log_path=False, emoji=True)

router = APIRouter(prefix="/research", tags=["Research"])

# =========================
# Enrichment + Discovery Config
# =========================
import httpx

OPENALEX_BASE = "https://api.openalex.org/works"
CROSSREF_BASE = "https://api.crossref.org/works"

# Single contact email for UA + OpenAlex mailto
CONTACT_EMAIL = "rushilvmehta999@gmail.com"

# Absolute path for SJR lookup file (update for prod if needed)
SJR_LOOKUP_FILE = Path(r"D:\D\AI\clg\i-smart-backend\app\routers\sjr_lookup_claudeSjrQ.json")
sjr_lookup_data: Dict[str, Dict[str, Any]] = {}
_sjr_loaded = False

# --- Global rate limiting controls ---
# Per-host concurrency (extra safety on top of batch size)
OPENALEX_CONCURRENCY = 2
CROSSREF_CONCURRENCY = 4

# Soft delay for Crossref (OpenAlex uses strict limiter below)
CROSSREF_MIN_DELAY = 0.25

# Retries / backoffs
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.2  # seconds

# Semaphores (per API host)
sem_openalex = asyncio.Semaphore(OPENALEX_CONCURRENCY)
sem_crossref = asyncio.Semaphore(CROSSREF_CONCURRENCY)

# Timestamp of last call (for Crossref)
last_request_time = {"crossref": None}

# STRICT global limiter for OpenAlex: <= ~9 req/sec
openalex_lock = asyncio.Lock()
_last_openalex_call: Optional[datetime] = None
OPENALEX_TARGET_INTERVAL = 0.11  # ~9 r/s

# =========================
# Tiny WS Notification Manager
# =========================
class WSNotifier:
    """In-memory connection manager keyed by user_id (e.g., email)."""
    def __init__(self) -> None:
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, user_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.setdefault(user_id, set()).add(ws)
        console.log(f":electric_plug: WS connected for user_id={user_id}")

    async def disconnect(self, user_id: str, ws: WebSocket) -> None:
        async with self._lock:
            try:
                self._connections.get(user_id, set()).discard(ws)
                if not self._connections.get(user_id):
                    self._connections.pop(user_id, None)
            except KeyError:
                pass
        console.log(f":electric_plug: WS disconnected for user_id={user_id}")

    async def send_event(self, user_id: str, event: str, payload: Dict[str, Any]) -> None:
        async with self._lock:
            conns = list(self._connections.get(user_id, set()))
        if not conns:
            console.log(f"[yellow]No WS clients for user_id={user_id}; skipping event {event}[/]")
            return
        message = {"event": event, "payload": payload, "ts": datetime.utcnow().isoformat() + "Z"}
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                # Best-effort; drop broken sockets
                await self.disconnect(user_id, ws)

ws_notifier = WSNotifier()

@router.websocket("/ws/{user_id}")
async def research_notifications_ws(websocket: WebSocket, user_id: str):
    """Simple WS endpoint: frontend connects here to receive ingest notifications."""
    try:
        await ws_notifier.connect(user_id, websocket)
        while True:
            # Keep alive; we don't expect incoming messages, but we can receive pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_notifier.disconnect(user_id, websocket)
    except Exception:
        await ws_notifier.disconnect(user_id, websocket)

# ---------- Helpers ----------
def _extract_keywords_from_project(project: Project) -> List[str]:
    """
    Reads project.expanded_query (JSON) and returns keywords list.
    Fallback to raw_query (comma/line separated) if no JSON/keywords found.
    """
    # prefer expanded_query
    if project.expanded_query:
        try:
            data = json.loads(project.expanded_query)
            if isinstance(data, dict) and isinstance(data.get("keywords"), list):
                kws = [str(k).strip() for k in data["keywords"] if str(k).strip()]
                if kws:
                    return kws
        except Exception:
            pass

    # fallback: raw_query as CSV/lines
    if project.raw_query:
        raw = project.raw_query.strip()
        if raw:
            parts = [p.strip() for p in raw.replace("\r", "\n").split("\n")]
            out: List[str] = []
            for block in parts:
                out.extend([x.strip() for x in block.split(",") if x.strip()])
            if out:
                return out

    return []


def ensure_sjr_loaded() -> None:
    """Lazy-load the SJR lookup once (works even if router startup hooks aren't firing)."""
    global _sjr_loaded, sjr_lookup_data
    if _sjr_loaded:
        return
    try:
        with open(SJR_LOOKUP_FILE, "r", encoding="utf-8") as f:
            sjr_lookup_data = json.load(f)
        console.log(f":white_check_mark: Loaded {len(sjr_lookup_data)} SJR records from {SJR_LOOKUP_FILE}")
    except FileNotFoundError:
        console.log(f":warning: SJR file not found at {SJR_LOOKUP_FILE} (ranking limited).")
        sjr_lookup_data = {}
    except json.JSONDecodeError:
        console.log(f":warning: Invalid JSON in {SJR_LOOKUP_FILE} (ranking limited).")
        sjr_lookup_data = {}
    _sjr_loaded = True


def clean_issn(issn: Optional[str]) -> str:
    return re.sub(r"\D", "", issn or "")


def strip_html(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = re.sub(r"<[^>]+>", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


def deconstruct_abstract(inverted_index: Optional[Dict]) -> Optional[str]:
    if not inverted_index:
        return None
    try:
        valid_indices = [val[0] for val in inverted_index.values()
                         if isinstance(val, list) and val]
        if not valid_indices:
            return None
        word_list = [""] * (max(valid_indices) + 1)
        for word, indices in inverted_index.items():
            if isinstance(indices, list):
                for i in indices:
                    if isinstance(i, int) and 0 <= i < len(word_list):
                        word_list[i] = word
        return " ".join(w for w in word_list if w).strip()
    except Exception:
        return None


async def _api_wait(api: str, base_delay: float = 0.0):
    """
    Rate-limit helper.
    - For OpenAlex: strict global <= ~9 r/s using lock + interval.
    - For Crossref: soft per-host spacing.
    """
    global _last_openalex_call, last_request_time
    now = datetime.now()

    if api == "openalex":
        async with openalex_lock:
            if _last_openalex_call is not None:
                elapsed = (now - _last_openalex_call).total_seconds()
                need = OPENALEX_TARGET_INTERVAL - elapsed
                if need > 0:
                    # tiny jitter makes parallel callers less bursty
                    await asyncio.sleep(need + random.uniform(0.005, 0.02))
            _last_openalex_call = datetime.now()
        return

    # Crossref soft limiter
    last = last_request_time.get(api)
    if last:
        elapsed = (now - last).total_seconds()
        need = base_delay - elapsed
        if need > 0:
            await asyncio.sleep(need)
    last_request_time[api] = datetime.now()


def calculate_quality_score(p: Dict[str, Any]) -> float:
    score = 0.0
    sjr = p.get("sjrScore")
    if isinstance(sjr, (int, float)):
        score += min(40.0, 40.0 * (1 - math.exp(-sjr / 3.0)))
    h = p.get("h_index")
    if isinstance(h, (int, float)):
        score += min(30.0, 30.0 * (min(h, 200) / 200.0))
    qmap = {"Q1": 20, "Q2": 15, "Q3": 10, "Q4": 5}
    q = p.get("quartile")
    if isinstance(q, str):
        score += qmap.get(q.upper(), 0)
    cites = p.get("citationCount")
    if isinstance(cites, int) and cites > 0:
        score += min(30.0, math.log10(cites + 1) * 10.0)
    year = p.get("yearPublished")
    if isinstance(year, int):
        years_old = datetime.now().year - year
        if years_old <= 5:
            score += max(10 - (years_old * 2), 0)
    if p.get("isFreelyAvailable"):
        score += 5
    if p.get("abstract"):
        score += 5
    return round(score, 2)


def normalize_query(q: str) -> str:
    tokens = [t.strip() for t in re.split(r"[,;\s]+", q or "") if t.strip()]
    seen, out = set(), []
    for t in tokens:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return " ".join(out)


# ---------- External calls: OpenAlex ----------
async def openalex_fetch_pages(client: httpx.AsyncClient, norm_query: str, pages: int = 4) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Cursor-based fetching with robust 403/429 handling.
    Returns (results, had_hard_block).
    """
    results: List[Dict[str, Any]] = []
    cursor = "*"
    per_page_options = [50, 25, 10, 5]  # adaptive when throttled
    per_page_idx = 0
    page_no = 0
    had_hard_block = False

    while page_no < pages:
        params = {
            "filter": f"default.search:{norm_query},has_doi:true",
            "per_page": per_page_options[per_page_idx],
            "sort": "relevance_score:desc",
            "select": "id,doi,title,authorships,cited_by_count,publication_year,open_access,abstract_inverted_index,host_venue",
            "cursor": cursor,
            "mailto": CONTACT_EMAIL,  # OpenAlex recommends including this
        }
        try:
            await _api_wait("openalex")  # strict global limiter
            async with sem_openalex:
                r = await client.get(OPENALEX_BASE, params=params, timeout=25.0)
            r.raise_for_status()
            payload = r.json()
            batch = payload.get("results", []) or []
            if not batch:
                break
            results.extend(batch)
            next_cursor = (payload.get("meta") or {}).get("next_cursor")
            if not next_cursor:
                break
            cursor = next_cursor
            page_no += 1

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            # Backoff + shrink page size on 403/429, then retry same page
            if code in (403, 429):
                if per_page_idx < len(per_page_options) - 1:
                    per_page_idx += 1
                    delay = INITIAL_BACKOFF * (per_page_idx + 1)
                    console.log(f"[yellow]OpenAlex {code} → retry per_page={per_page_options[per_page_idx]} after {delay:.1f}s[/]")
                    await asyncio.sleep(delay + random.uniform(0.1, 0.3))
                    continue
                else:
                    console.log(f"[red]OpenAlex hard {code}[/]: {e}")
                    had_hard_block = True
                    break
            # Other codes: stop
            console.log(f"[yellow]OpenAlex HTTP {code}[/]: {e}")
            break

        except Exception as ex:
            console.log(f"[red]OpenAlex error[/]: {ex}")
            break

    return results, had_hard_block


# ---------- External calls: Crossref (fallback discovery & DOI enrich) ----------
async def enrich_with_crossref(client: httpx.AsyncClient, doi: Optional[str]) -> Optional[Dict[str, Any]]:
    if not doi:
        return None
    url = f"{CROSSREF_BASE}/{doi}"
    delay = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            await _api_wait("crossref", base_delay=CROSSREF_MIN_DELAY)
            async with sem_crossref:
                r = await client.get(url, timeout=15.0)
            r.raise_for_status()
            return r.json().get("message", {})
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = delay * (attempt + 1)
                await asyncio.sleep(wait)
                delay *= 1.5
                continue
            if e.response.status_code == 404:
                return None
            return None
        except (httpx.RequestError, ValueError):
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay)
                delay *= 1.5
                continue
            return None
        except Exception:
            return None
    return None


async def crossref_search(client: httpx.AsyncClient, query: str, rows: int = 40, pages: int = 2) -> List[Dict[str, Any]]:
    """Fallback search via Crossref when OpenAlex is blocked."""
    results: List[Dict[str, Any]] = []
    offset = 0
    for _ in range(pages):
        params = {
            "query.bibliographic": query,
            "filter": "has-full-text:true",
            "rows": rows,
            "offset": offset,
        }
        try:
            await _api_wait("crossref", base_delay=CROSSREF_MIN_DELAY)
            async with sem_crossref:
                r = await client.get(CROSSREF_BASE, params=params, timeout=25.0)
            r.raise_for_status()
            data = r.json().get("message", {}) or {}
            items = data.get("items", []) or []
            if not items:
                break
            results.extend(items)
            offset += rows
        except Exception as e:
            console.log(f"[yellow]Crossref search error[/]: {e}")
            break
    return results


# ---------- One-keyword pipeline (OpenAlex primary, Crossref fallback) ----------
async def discover_and_process(query: str, fetch_pages: int = 3, openalex_enabled: bool = True, crossref_enabled: bool = True) -> List[Dict[str, Any]]:
    """
    Run discovery for one keyword, enrich with Crossref + SJR, score, and return.
    """
    ensure_sjr_loaded()
    out: List[Dict[str, Any]] = []
    norm_query = normalize_query(query)

    headers = {
        "User-Agent": f"iSMART-API/2.0 (mailto:{CONTACT_EMAIL})",
        "Accept": "application/json",
    }

    limits = httpx.Limits(max_connections=12, max_keepalive_connections=12)
    async with httpx.AsyncClient(headers=headers, timeout=30.0, limits=limits) as client:
        works = []
        hard_block = False

        if openalex_enabled:
            works, hard_block = await openalex_fetch_pages(client, norm_query, pages=fetch_pages)

        if works:
            # Crossref enrich concurrently for DOI metadata (title/container/ISSN/abstract)
            tasks = [enrich_with_crossref(client, w.get("doi")) for w in works]
            crossref_meta = await asyncio.gather(*tasks, return_exceptions=True)

            for w, meta in zip(works, crossref_meta):
                if not w:
                    continue

                # core fields from OpenAlex
                title = w.get("title")
                authors = [
                    a["author"]["display_name"]
                    for a in w.get("authorships", [])
                    if a.get("author")
                ]
                doi = w.get("doi")
                citation_count = w.get("cited_by_count", 0)
                year = w.get("publication_year")
                abstract = deconstruct_abstract(w.get("abstract_inverted_index"))
                oa = w.get("open_access", {}) or {}
                is_oa = oa.get("is_oa", False)
                oa_url = oa.get("oa_url")

                journal_title = None
                issn_candidates: List[str] = []

                # Crossref data (if available)
                if meta and not isinstance(meta, Exception):
                    ct = meta.get("container-title", []) or []
                    journal_title = ct[0] if ct else None
                    issn_candidates.extend(meta.get("ISSN", []) or [])
                    if not abstract:
                        abstract = strip_html(meta.get("abstract"))

                # fallback to OpenAlex host_venue
                host_venue = w.get("host_venue") or {}
                if host_venue:
                    hv_issns = host_venue.get("issn") or []
                    if isinstance(hv_issns, str):
                        hv_issns = [hv_issns]
                    issn_l = host_venue.get("issn_l")
                    if issn_l:
                        issn_candidates.append(issn_l)
                    issn_candidates.extend(hv_issns)
                    if not journal_title:
                        journal_title = host_venue.get("display_name")

                # choose an ISSN
                issn_clean = None
                for raw in issn_candidates:
                    c = clean_issn(raw)
                    if c:
                        issn_clean = c
                        break

                quartile = h_index = sjr_score = None
                if issn_clean and issn_clean in sjr_lookup_data:
                    r = sjr_lookup_data[issn_clean]
                    quartile = r.get("quartile")
                    h_index = r.get("h_index")
                    sjr_score = r.get("sjr")

                paper = {
                    "title": title,
                    "authors": authors,
                    "doi": doi,
                    "citationCount": citation_count,
                    "yearPublished": year,
                    "abstract": abstract,
                    "isFreelyAvailable": is_oa,
                    "downloadUrl": oa_url,
                    "journalTitle": journal_title,
                    "issn": issn_clean,
                    "quartile": quartile,
                    "h_index": h_index,
                    "sjrScore": sjr_score,
                }
                paper["qualityScore"] = calculate_quality_score(paper)
                out.append(paper)

        # Fallback (or disabled OpenAlex)
        if (not works) and crossref_enabled:
            if hard_block:
                console.log(f"[yellow]Switching to Crossref fallback due to OpenAlex block for keyword[/] '{query}'")
            cr_items = await crossref_search(client, norm_query, rows=40, pages=max(1, fetch_pages))
            for it in cr_items:
                doi = it.get("DOI")
                title_list = it.get("title") or []
                title = title_list[0] if title_list else None
                author_objs = it.get("author") or []
                authors = []
                for a in author_objs:
                    name = a.get("name") or (" ".join([a.get("given") or "", a.get("family") or ""]).strip() or None)
                    if name:
                        authors.append(name)

                citation_count = it.get("is-referenced-by-count", 0)
                year = None
                issued = it.get("issued", {}) or {}
                date_parts = issued.get("date-parts") or []
                if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
                    year = date_parts[0][0]

                abstract = strip_html(it.get("abstract"))
                is_oa = bool(it.get("license") or it.get("link"))
                oa_url = None
                links = it.get("link") or []
                if links:
                    oa_url = links[0].get("URL") or None

                journal_title = None
                ct = it.get("container-title") or []
                if ct:
                    journal_title = ct[0]
                issn_candidates = it.get("ISSN") or []
                issn_clean = None
                for raw in issn_candidates:
                    c = clean_issn(raw)
                    if c:
                        issn_clean = c
                        break

                quartile = h_index = sjr_score = None
                if issn_clean and issn_clean in sjr_lookup_data:
                    r = sjr_lookup_data[issn_clean]
                    quartile = r.get("quartile")
                    h_index = r.get("h_index")
                    sjr_score = r.get("sjr")

                paper = {
                    "title": title,
                    "authors": authors,
                    "doi": doi,
                    "citationCount": citation_count,
                    "yearPublished": year,
                    "abstract": abstract,
                    "isFreelyAvailable": is_oa,
                    "downloadUrl": oa_url,
                    "journalTitle": journal_title,
                    "issn": issn_clean,
                    "quartile": quartile,
                    "h_index": h_index,
                    "sjrScore": sjr_score,
                }
                paper["qualityScore"] = calculate_quality_score(paper)
                out.append(paper)

    return out


# ---------- Batched multi-keyword orchestrator (safe parallel) ----------
async def discover_many_keywords_batched(
    keywords: List[str],
    fetch_pages_per_keyword: int = 2,
    batch_size: int = 3,
    inter_batch_delay_ms: int = 800,
    openalex_enabled: bool = True,
    crossref_enabled: bool = True,
) -> List[Dict[str, Any]]:

    console.log(f"Discover across {len(keywords)} keywords "
                f"(pages_per_keyword={fetch_pages_per_keyword}, batch_size={batch_size})")

    merged: List[Dict[str, Any]] = []
    seen_doi = set()
    seen_title = set()

    # chunk keywords
    for start in range(0, len(keywords), batch_size):
        batch = keywords[start: start + batch_size]
        console.log(f"[blue]Batch {start//batch_size + 1}[/] starting for {len(batch)} keyword(s): {batch}")

        # run this batch concurrently, but with the internal per-API semaphores & waits
        tasks = [
            discover_and_process(
                kw,
                fetch_pages=fetch_pages_per_keyword,
                openalex_enabled=openalex_enabled,
                crossref_enabled=crossref_enabled,
            )
            for kw in batch
        ]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # merge, guarding errors per keyword
        for kw, res in zip(batch, results_lists):
            if isinstance(res, Exception):
                console.log(f"[red]Keyword pipeline failed[/] '{kw}': {res}")
                continue
            for p in res:
                doi = (p.get("doi") or "").lower().strip()
                title = (p.get("title") or "").lower().strip()
                if doi and doi in seen_doi:
                    continue
                if (not doi) and title and title in seen_title:
                    continue
                if doi:
                    seen_doi.add(doi)
                elif title:
                    seen_title.add(title)
                merged.append(p)

        console.log(f"[green]Batch merged[/]: total so far {len(merged)} papers")

        # cool down between batches
        if start + batch_size < len(keywords):
            delay_s = max(0, inter_batch_delay_ms) / 1000.0
            await asyncio.sleep(delay_s)

    console.log(f"Merged across batches → {len(merged)} unique papers")
    return merged


# ---------- Background authors processing ----------
def _create_and_link_authors(session: Session, paper_id: int, author_names: List[str]) -> Tuple[int, int]:
    created_authors = 0
    linked_authors = 0

    for author_name in (author_names or []):
        author_affil = None
        author = (
            session.query(Author)
            .filter(Author.name == author_name, Author.affiliation == author_affil)
            .first()
        )
        if author is None:
            author = Author(name=author_name, affiliation=author_affil)
            session.add(author)
            try:
                session.flush()
                created_authors += 1
            except IntegrityError:
                session.rollback()
                author = (
                    session.query(Author)
                    .filter(Author.name == author_name, Author.affiliation == author_affil)
                    .first()
                )
                if author is None:
                    raise

        exists = (
            session.query(PaperAuthor)
            .filter(PaperAuthor.paper_id == paper_id, PaperAuthor.author_id == author.author_id)
            .first()
        )
        if exists is None:
            session.add(PaperAuthor(paper_id=paper_id, author_id=author.author_id))
            linked_authors += 1

    return created_authors, linked_authors


def _process_authors_background(paper_authors_map: Dict[int, List[str]], notify_user: Optional[str]) -> None:
    """Runs in BackgroundTasks after the main request finishes."""
    if SessionLocal is None:
        console.log("[yellow]SessionLocal unavailable; cannot process authors in background.[/]")
        return

    session = SessionLocal()
    total_created = 0
    total_linked = 0
    try:
        for paper_id, names in paper_authors_map.items():
            c, l = _create_and_link_authors(session, paper_id, names)
            total_created += c
            total_linked += l
        session.commit()
        console.log(f":busts_in_silhouette: [green]Background authors processed[/] created={total_created}, linked={total_linked}")
    except Exception as e:
        session.rollback()
        console.log(f"[red]Background author processing failed[/]: {e}")
    finally:
        session.close()

    # Notify via WS (best-effort, fire-and-forget via asyncio.create_task)
    if notify_user:
        asyncio.create_task(ws_notifier.send_event(
            notify_user,
            "authors_processed",
            {"created_authors": total_created, "linked_authors": total_linked}
        ))


# ------------------ API: Ingest & Save with filters + rich logs ------------------
@router.post("/ingest/{project_id}")
async def ingest_papers_for_project(
    project_id: int,
    limit: int = Query(25, ge=1, le=200, description="Cap number of papers to save"),
    save_recommendations: bool = Query(True, description="Also save ranking as recommendations"),
    pages_per_keyword: int = Query(2, ge=1, le=5, description="Pages per keyword per provider"),

    # Safe-parallelism controls:
    keyword_batch_size: int = Query(3, ge=1, le=10, description="How many keywords to process concurrently"),
    inter_batch_delay_ms: int = Query(800, ge=0, le=10000, description="Cooldown between keyword batches (ms)"),
    openalex_enabled: bool = Query(True, description="Use OpenAlex as primary source"),
    crossref_enabled: bool = Query(True, description="Allow Crossref fallback/primary"),

    # Filters (optional)
    min_citations: Optional[int] = Query(None, ge=0, description="Only include papers with at least this many citations"),
    year_min: Optional[int] = Query(None, description="Minimum publication year (inclusive)"),
    year_max: Optional[int] = Query(None, description="Maximum publication year (inclusive)"),
    only_open_access: bool = Query(False, description="Only include open-access papers"),
    quartile_in: Optional[str] = Query(None, description="Comma-separated list of allowed SJR quartiles (e.g., 'Q1,Q2')"),
    require_abstract: bool = Query(False, description="Skip papers with missing abstract"),

    # Notify + background authors
    notify_user: Optional[str] = Query("rushilvmehta999@gmail.com", description="User ID (e.g., email) to notify via WS"),
    authors_in_background: bool = Query(True, description="Process authors in background after commit"),

    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None,
):
    """
    Pull keywords, discover via batched safe-parallel pipelines (OpenAlex→Crossref),
    enrich with SJR, rank, filter, and persist.
    Notifies via WebSocket after commit; authors can be processed in background.
    """
    console.rule(f":mag: [bold]Ingest Papers[/] • Project ID: [cyan]{project_id}[/]")

    ensure_sjr_loaded()

    # Fetch project
    try:
        project: Optional[Project] = db.query(Project).filter(Project.project_id == project_id).first()
    except Exception as e:
        console.log(f"[bold red]DB error while fetching project {project_id}[/]: {e}")
        raise

    if not project:
        console.log(f"[bold red]Project not found[/]: {project_id}")
        raise HTTPException(status_code=404, detail="Project not found")

    console.log("[green]Loaded project[/]", Pretty({
        "project_id": project.project_id,
        "name": getattr(project, "name", None),
    }, expand_all=True))

    # Extract keywords
    keywords = _extract_keywords_from_project(project)
    if not keywords:
        console.log("[bold red]No keywords found in project (expanded_query.keywords or raw_query).[/]")
        raise HTTPException(
            status_code=400,
            detail="No keywords found in project (expanded_query.keywords or raw_query).",
        )

    # Show keywords table
    kw_table = Table(title="Keywords", show_header=True, header_style="bold magenta")
    kw_table.add_column("#", justify="right")
    kw_table.add_column("Keyword", overflow="fold")
    for i, kw in enumerate(keywords, start=1):
        kw_table.add_row(str(i), kw)
    console.print(kw_table)

    # Discover & enrich (batched safe-parallel)
    console.log(f"Discovering (pages_per_keyword={pages_per_keyword}, batch_size={keyword_batch_size}) …")
    discovered = await discover_many_keywords_batched(
        keywords,
        fetch_pages_per_keyword=pages_per_keyword,
        batch_size=keyword_batch_size,
        inter_batch_delay_ms=inter_batch_delay_ms,
        openalex_enabled=openalex_enabled,
        crossref_enabled=crossref_enabled,
    )

    if not discovered:
        console.log("[yellow]No papers discovered for the given keywords (OpenAlex+Crossref).[/]")
        raise HTTPException(status_code=404, detail="No papers discovered for the given keywords.")

    # Apply filters (only if provided)
    allowed_quartiles = None
    if quartile_in:
        allowed_quartiles = {q.strip().upper() for q in quartile_in.split(",") if q.strip()}

    before_filter = len(discovered)

    def _pass_filters(p: Dict[str, Any]) -> bool:
        if min_citations is not None and int(p.get("citationCount") or 0) < min_citations:
            return False
        y = p.get("yearPublished")
        if year_min is not None and isinstance(y, int) and y < year_min:
            return False
        if year_max is not None and isinstance(y, int) and y > year_max:
            return False
        if only_open_access and not p.get("isFreelyAvailable"):
            return False
        if allowed_quartiles and (str(p.get("quartile") or "").upper() not in allowed_quartiles):
            return False
        if require_abstract and not p.get("abstract"):
            return False
        return True

    discovered = [p for p in discovered if _pass_filters(p)]
    console.log(f"Filtered {before_filter} → {len(discovered)} using params", Pretty({
        "min_citations": min_citations,
        "year_min": year_min,
        "year_max": year_max,
        "only_open_access": only_open_access,
        "quartile_in": list(allowed_quartiles) if allowed_quartiles else None,
        "require_abstract": require_abstract,
    }))

    if not discovered:
        console.log("[yellow]All papers were filtered out by the chosen filters.[/]")
        raise HTTPException(status_code=404, detail="No papers remain after filters.")

    # sort by quality descending and cap
    discovered.sort(key=lambda p: p.get("qualityScore", 0), reverse=True)
    total_before_cap = len(discovered)
    discovered = discovered[:limit]
    console.log(f"Discovered {total_before_cap} papers after filters; capping to [bold]{len(discovered)}[/].")

    # Preview table
    pre_table = Table(title="Selected Papers (pre-save)", show_lines=False, header_style="bold blue")
    pre_table.add_column("Rank", justify="right")
    pre_table.add_column("Title", overflow="fold", max_width=80)
    pre_table.add_column("DOI", overflow="fold", max_width=40)
    pre_table.add_column("Year", justify="center")
    pre_table.add_column("Journal", overflow="fold", max_width=40)
    pre_table.add_column("Quartile", justify="center")
    pre_table.add_column("SJR", justify="right")
    pre_table.add_column("h-index", justify="right")
    pre_table.add_column("Cites", justify="right")
    pre_table.add_column("Quality", justify="right")

    for rank, p in enumerate(discovered, start=1):
        pre_table.add_row(
            str(rank),
            (p.get("title") or "Untitled"),
            (p.get("doi") or "—"),
            str(p.get("yearPublished") or "—"),
            (p.get("journalTitle") or "—"),
            str(p.get("quartile") or "—"),
            f"{p.get('sjrScore'):.3f}" if isinstance(p.get("sjrScore"), (int, float)) else "—",
            str(p.get("h_index") or "—"),
            str(p.get("citationCount") or 0),
            f'{p.get("qualityScore", 0):.2f}',
        )
    console.print(pre_table)

    # Save
    saved_count = 0
    created_authors = 0
    linked_authors = 0
    created_recs = 0
    upserted_papers: List[int] = []
    paper_authors_map: Dict[int, List[str]] = {}

    console.rule("[bold]Upsert & Link[/]")

    for rank, p in enumerate(discovered, start=1):
        doi = p.get("doi")
        title = p.get("title") or "Untitled"

        # Upsert paper by DOI (unique) or by title if DOI missing
        paper_obj: Optional[Paper] = None
        if doi:
            paper_obj = db.query(Paper).filter(Paper.doi == doi).first()
        if paper_obj is None and not doi:
            paper_obj = db.query(Paper).filter(Paper.title == title).first()

        if paper_obj is None:
            paper_obj = Paper(
                query_id=project.project_id,
                title=title,
                abstract=p.get("abstract"),
                doi=doi,
                url=p.get("downloadUrl"),
                publication_year=p.get("yearPublished"),
                journal=p.get("journalTitle"),
                paper_type=None,
                citation_count=p.get("citationCount"),
                # store SJR in impact_factor column (as you noted)
                impact_factor=p.get("sjrScore"),
                fetched_from="openalex",
                ingestion_date=datetime.utcnow(),
            )
            # optional fields if your model has them
            for attr, val in [
                ("issn", p.get("issn")),
                ("sjr_quartile", p.get("quartile")),
                ("h_index", p.get("h_index")),
                ("is_open_access", p.get("isFreelyAvailable")),
                ("oa_url", p.get("downloadUrl")),
            ]:
                if hasattr(Paper, attr):
                    setattr(paper_obj, attr, val)

            db.add(paper_obj)
            try:
                db.flush()
                console.log(f":new: [bold green]Created Paper[/] #{paper_obj.paper_id} • Rank {rank} • {title}")
            except IntegrityError:
                db.rollback()
                paper_obj = db.query(Paper).filter(Paper.doi == doi).first() if doi else None
                if paper_obj is None:
                    console.log(f"[bold red]IntegrityError upserting paper (non-recoverable)[/]: {title}")
                    raise
                else:
                    console.log(f"[yellow]Recovered existing paper after IntegrityError[/]: id={paper_obj.paper_id}")
        else:
            # update selective fields
            before_id = paper_obj.paper_id
            paper_obj.title = title or paper_obj.title
            paper_obj.abstract = p.get("abstract") or paper_obj.abstract
            paper_obj.url = p.get("downloadUrl") or paper_obj.url
            paper_obj.publication_year = p.get("yearPublished") or paper_obj.publication_year
            paper_obj.journal = p.get("journalTitle") or paper_obj.journal
            paper_obj.citation_count = p.get("citationCount") or paper_obj.citation_count
            paper_obj.impact_factor = p.get("sjrScore") or paper_obj.impact_factor
            paper_obj.fetched_from = paper_obj.fetched_from or "openalex"
            # optional fields if columns exist
            for attr, val in [
                ("issn", p.get("issn")),
                ("sjr_quartile", p.get("quartile")),
                ("h_index", p.get("h_index")),
                ("is_open_access", p.get("isFreelyAvailable")),
                ("oa_url", p.get("downloadUrl")),
            ]:
                if hasattr(Paper, attr) and val is not None:
                    setattr(paper_obj, attr, val)

            console.log(f":pencil: [cyan]Updated Paper[/] #{before_id} • Rank {rank} • {title}")

        upserted_papers.append(paper_obj.paper_id)

        # Authors: collect to process later (background) or do inline
        author_names = p.get("authors") or []
        if authors_in_background:
            paper_authors_map[paper_obj.paper_id] = author_names
        else:
            c, l = _create_and_link_authors(db, paper_obj.paper_id, author_names)
            created_authors += c
            linked_authors += l
            if c or l:
                console.log(f"    :bust_in_silhouette: authors created={c}, linked={l}")

        # Recommendations (optional)
        if save_recommendations:
            rec_exists = (
                db.query(Recommendation)
                .filter(Recommendation.query_id == project.project_id, Recommendation.paper_id == paper_obj.paper_id)
                .first()
            )
            if rec_exists is None:
                rec = Recommendation(
                    query_id=project.project_id,
                    paper_id=paper_obj.paper_id,
                    relevance_score=p.get("qualityScore") or 0.0,
                    novelty_score=None,
                    overall_rank=rank,
                    recommended_reason="Auto-ranked by composite quality score",
                )
                db.add(rec)
                created_recs += 1
                console.log(f"    :bookmark_tabs: [green]Created Recommendation[/] rank={rank} score={p.get('qualityScore') or 0.0:.3f}")

        saved_count += 1

    # Commit once
    console.rule("[bold]Commit[/]")
    try:
        db.commit()
        console.log("[bold green]Transaction committed successfully[/] :white_check_mark:")
    except IntegrityError as e:
        db.rollback()
        console.log("[bold red]DB integrity error during commit[/]", e)
        raise HTTPException(status_code=500, detail=f"DB integrity error: {str(e)}")
    except Exception as e:
        db.rollback()
        console.log("[bold red]DB error during commit[/]", e)
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)}")

    # Schedule background authors (if requested)
    if authors_in_background and paper_authors_map:
        if background_tasks is not None and SessionLocal is not None:
            background_tasks.add_task(_process_authors_background, paper_authors_map, notify_user)
            console.log(":rocket: [green]Scheduled background author processing[/]")
        else:
            console.log("[yellow]BackgroundTasks or SessionLocal unavailable; authors not processed in background.[/]")

    # Notify via WebSocket (immediately after commit)
    if notify_user:
        await ws_notifier.send_event(
            notify_user,
            "ingest_complete",
            {
                "project_id": project.project_id,
                "keywords_used": len(keywords),
                "papers_saved_or_updated": saved_count,
                "paper_ids": upserted_papers,
                "recommendations_created": created_recs,
                "authors_mode": "background" if authors_in_background else "inline",
            },
        )

    # final summary to logs
    summary = Table(title="Ingestion Summary", header_style="bold white")
    summary.add_column("Field", justify="left", style="cyan")
    summary.add_column("Value", justify="right", style="bold")

    summary.add_row("Project ID", str(project.project_id))
    summary.add_row("Keywords Used", str(len(keywords)))
    summary.add_row("Papers Saved/Updated", str(saved_count))
    summary.add_row("Paper IDs", ", ".join(str(i) for i in upserted_papers) if upserted_papers else "—")
    summary.add_row("Authors Created (inline)", str(created_authors))
    summary.add_row("Author Links Created (inline)", str(linked_authors))
    summary.add_row("Recommendations Created", str(created_recs))
    summary.add_row("Authors Mode", "background" if authors_in_background else "inline")
    summary.add_row("WS Notify User", notify_user or "—")

    console.print(Panel(summary, title=":clipboard: Result", border_style="green"))
    console.rule("[dim]End Ingest[/]")

    # API response
    return {
        "project_id": project.project_id,
        "keywords_used": keywords,
        "papers_saved_or_updated": saved_count,
        "paper_ids": upserted_papers,
        "authors_created_inline": created_authors,
        "author_links_created_inline": linked_authors,
        "recommendations_created": created_recs,
        "filters": {
            "min_citations": min_citations,
            "year_min": year_min,
            "year_max": year_max,
            "only_open_access": only_open_access,
            "quartile_in": list(allowed_quartiles) if allowed_quartiles else None,
            "require_abstract": require_abstract,
        },
        "batching": {
            "keyword_batch_size": keyword_batch_size,
            "inter_batch_delay_ms": inter_batch_delay_ms,
            "openalex_enabled": openalex_enabled,
            "crossref_enabled": crossref_enabled,
        },
        "authors_mode": "background" if authors_in_background else "inline",
        "ws_notified": bool(notify_user),
    }
# ------------------ API: Fetch Papers for a Project ------------------
