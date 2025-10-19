# app/api/literature.py
import os
import re
import uuid
import shutil
import json
from typing import Dict, Optional, List, Any

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

from core.database import get_db
from model.models import Project, Paper, PaperAnalysis
from llm.llm import LLMClient

# rich for console printing
from rich.console import Console
from rich.json import JSON as RichJSON

console = Console()
router = APIRouter(prefix="/literature", tags=["Literature Review"])

UPLOAD_ROOT = os.environ.get("UPLOAD_ROOT", "./uploads_papers")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

ALLOWED_EXTS = {".pdf", ".docx", ".txt"}


# ---------------------------
# Basic file/text helpers
# ---------------------------
def _safe_ext(name: str) -> str:
    return os.path.splitext(name.lower())[1]


def extract_pdf_metadata(file_path: str) -> Dict[str, Optional[str]]:
    try:
        reader = PdfReader(file_path)
        info = reader.metadata or {}
        author = getattr(info, "author", None) or info.get("/Author") or info.get("Author")
        title = getattr(info, "title", None) or info.get("/Title") or info.get("Title")
        year = None
        raw_date = getattr(info, "creation_date", None) or info.get("/CreationDate") or info.get("CreationDate")
        if raw_date:
            m = re.search(r"(\d{4})", str(raw_date))
            if m:
                year = m.group(1)
        return {"author": author, "title": title, "year": year}
    except Exception:
        return {"author": None, "title": None, "year": None}


def load_pdf_text(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    if not docs:
        raise ValueError("No content extracted from PDF.")
    return "\n".join(f"[Page {i+1}]\n{d.page_content}" for i, d in enumerate(docs))


def load_docx_text(file_path: str) -> str:
    doc = DocxDocument(file_path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if not paras:
        raise ValueError("No content extracted from DOCX.")
    return "\n".join(paras)


def load_txt_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        raise ValueError("No content extracted from TXT.")
    return text


def _extract_text_and_metadata(path: str) -> Dict[str, object]:
    """
    Extract full text and basic metadata from a supported file.
    """
    ext = _safe_ext(path)
    if ext == ".pdf":
        text = load_pdf_text(path)
        meta = extract_pdf_metadata(path)
    elif ext == ".docx":
        text = load_docx_text(path)
        meta = {"author": None, "title": None, "year": None}
    elif ext == ".txt":
        text = load_txt_text(path)
        meta = {"author": None, "title": None, "year": None}
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")
    return {"text": text, "metadata": meta}


# ---------------------------
# Pydantic schema for LLM output
# ---------------------------
class CitationSchema(BaseModel):
    author: str = "not reported"
    year: Any = "not reported"
    suggested_citation: str = "not reported"


class ReviewOutput(BaseModel):
    summary_text: str = Field(..., description="Concise summary (150-300 words) as plain text (no code fences).")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    peer_reviewed: Optional[bool] = None
    critique_score: Optional[float] = None
    tone: Optional[str] = None
    sentiment_score: Optional[float] = None
    semantic_patterns: List[str] = Field(default_factory=list)
    citation: CitationSchema = Field(default_factory=CitationSchema)

    @validator("critique_score")
    def check_critique_score(cls, v):
        if v is None:
            return None
        if not (0 <= v <= 10):
            raise ValueError("critique_score must be between 0 and 10")
        return v

    @validator("sentiment_score")
    def check_sentiment(cls, v):
        if v is None:
            return None
        if not (-1 <= v <= 1):
            raise ValueError("sentiment_score must be between -1 and 1")
        return v


# ---------------------------
# Prompt builder (short and strict)
# ---------------------------
def build_review_prompt(document_text: str, metadata: Dict[str, Optional[str]]) -> str:
    meta_hint = {
        "author": metadata.get("author") or "not reported",
        "title": metadata.get("title") or "not reported",
        "year": metadata.get("year") or "not reported",
    }

    prompt = (
        "You are a concise literature-review assistant. Use ONLY the provided document text.\n"
        "Produce a JSON object and NOTHING else. Do NOT wrap the JSON in markdown or triple-backticks.\n"
        "Make sure the JSON keys exactly include: summary_text, strengths, weaknesses, gaps,\n"
        "peer_reviewed, critique_score, tone, sentiment_score, semantic_patterns, citation.\n"
        "- summary_text must be a plain human-readable literature review (150-300 words). NO code fences.\n"
        "- strengths, weaknesses, gaps, semantic_patterns must be arrays of short strings.\n"
        "- peer_reviewed: true/false/null, critique_score: number 0-10 or null, sentiment_score: -1..1 or null.\n"
        "- citation must be an object: author, year, suggested_citation (use \"not reported\" if missing).\n"
        "If any info is missing, use null or the string \"not reported\". Keep all text concise.\n\n"
        f"Known metadata: {json.dumps(meta_hint)}\n\n"
        "---BEGIN_DOCUMENT---\n"
        f"{document_text}\n"
        "---END_DOCUMENT---\n"
    )
    return prompt


# ---------------------------
# Robust JSON extraction & cleanup (with nested JSON unwrapping)
# ---------------------------
def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text.strip()


def _find_first_json(s: str) -> Optional[str]:
    if not s:
        return None
    start_obj = s.find("{")
    start_arr = s.find("[")
    starts = [i for i in (start_obj, start_arr) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    # attempt progressive parse (bounded)
    max_len = min(len(s), start + 200000)
    for end in range(start + 1, max_len):
        candidate = s[start:end]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            continue
    # regex fallback
    m = re.search(r"(\{[\s\S]*\})", s)
    if m:
        return m.group(1)
    m2 = re.search(r"(\[[\s\S]*\])", s)
    if m2:
        return m2.group(1)
    return None


def _unwrap_json_strings(obj: Any, max_depth: int = 3) -> Any:
    """
    Recursively walk `obj` (dict/list/str) and attempt to parse any string that
    contains a JSON object/array. Limits recursion by max_depth to avoid loops.
    Returns a new structure with strings replaced by parsed JSON where possible.
    """
    if max_depth <= 0:
        return obj

    # If it's a dict, process each value
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _unwrap_json_strings(v, max_depth=max_depth - 1)
        return out

    # If it's a list, process each element
    if isinstance(obj, list):
        return [_unwrap_json_strings(v, max_depth=max_depth - 1) for v in obj]

    # If it's a string, try to parse it as JSON (safely)
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return obj
        # quick guard: must start with { or [ to even try
        if not (s.startswith("{") or s.startswith("[")):
            # try to find a JSON substring inside the string
            inner = _find_first_json(s)
            if inner:
                s = inner
            else:
                return obj
        # try parsing, with a small number of attempts (handle escaped JSON)
        for _ in range(3):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, (dict, list)):
                    return _unwrap_json_strings(parsed, max_depth=max_depth - 1)
                return parsed
            except Exception:
                try:
                    # attempt to unescape common escape sequences
                    s = s.encode("utf-8").decode("unicode_escape")
                except Exception:
                    break
        return obj

    # other primitives unchanged
    return obj


def _extract_json_from_llm(raw: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON dict from raw LLM output, and unwrap nested JSON-strings.
    """
    if raw is None:
        return {}

    raw_stripped = _strip_code_fences(raw)

    # 1) try direct JSON parse of cleaned text
    try:
        parsed = json.loads(raw_stripped)
        if isinstance(parsed, dict):
            return _unwrap_json_strings(parsed)
        if isinstance(parsed, list):
            return {"semantic_patterns": _unwrap_json_strings(parsed)}
    except Exception:
        pass

    # 2) find first JSON substring and attempt to parse
    candidate = _find_first_json(raw_stripped)
    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return _unwrap_json_strings(parsed)
            if isinstance(parsed, list):
                return {"semantic_patterns": _unwrap_json_strings(parsed)}
        except Exception:
            try:
                fixed = re.sub(r",\s*([\]\}])", r"\1", candidate)
                parsed = json.loads(fixed)
                if isinstance(parsed, dict):
                    return _unwrap_json_strings(parsed)
            except Exception:
                pass

    # 3) Try to salvage by searching any JSON-like substring and parse/unpack recursively
    try:
        m = re.search(r"(\{[\s\S]*\})", raw_stripped)
        if m:
            maybe = m.group(1)
            parsed = json.loads(maybe)
            if isinstance(parsed, dict):
                return _unwrap_json_strings(parsed)
    except Exception:
        pass

    # final fallback: treat the cleaned raw as the plain summary_text (no fences)
    return {"summary_text": raw_stripped}


def _serialize_list_for_db(lst: Optional[List[str]]) -> Optional[str]:
    if lst is None:
        return None
    try:
        return json.dumps(lst, ensure_ascii=False)
    except Exception:
        return json.dumps(list(map(str, lst)))


def _deserialize_list_from_db(s: Optional[str]) -> List[str]:
    if not s:
        return []
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return val
        return [str(val)]
    except Exception:
        return [x for x in s.splitlines() if x.strip()]


def _to_db_peer_reviewed(v: Optional[bool]) -> Optional[int]:
    if v is True:
        return 1
    if v is False:
        return 0
    return None


# ---------------------------
# Endpoints
# ---------------------------
@router.post("/project/{project_id}/upload-and-review")
async def upload_and_review_files(
    project_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found.")

    llm = LLMClient()
    project_dir = os.path.join(UPLOAD_ROOT, f"project_{project_id}")
    os.makedirs(project_dir, exist_ok=True)

    results = []
    for upload in files:
        file_name = getattr(upload, "filename", "unknown")
        try:
            # save file
            ext = _safe_ext(file_name)
            if ext not in ALLOWED_EXTS:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or 'unknown'}")
            unique_name = f"{uuid.uuid4().hex}_{os.path.basename(file_name)}"
            stored_path = os.path.join(project_dir, unique_name)
            with open(stored_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)

            # extract text & metadata
            extracted = _extract_text_and_metadata(stored_path)
            document_text = extracted["text"]
            metadata = extracted.get("metadata") or {}

            if not metadata.get("title"):
                metadata["title"] = file_name or os.path.splitext(unique_name)[0]

            prompt = build_review_prompt(document_text, metadata)

            # call LLM
            review_raw = llm.chat(prompt)
            if not review_raw or not review_raw.strip():
                raise RuntimeError("LLM returned an empty review.")

            review_json_candidate = _extract_json_from_llm(review_raw)

            minimal = {
                "summary_text": None,
                "strengths": [],
                "weaknesses": [],
                "gaps": [],
                "peer_reviewed": None,
                "critique_score": None,
                "tone": None,
                "sentiment_score": None,
                "semantic_patterns": [],
                "citation": {"author": "not reported", "year": "not reported", "suggested_citation": "not reported"},
            }
            merged = {**minimal, **(review_json_candidate or {})}

            # coerce boolean-like strings
            def _coerce_bool(x):
                if isinstance(x, bool):
                    return x
                if x is None:
                    return None
                if isinstance(x, str):
                    t = x.strip().lower()
                    if t in ("true", "yes", "1"):
                        return True
                    if t in ("false", "no", "0"):
                        return False
                return None

            if "peer_reviewed" in merged:
                merged["peer_reviewed"] = _coerce_bool(merged.get("peer_reviewed"))

            for k in ("critique_score", "sentiment_score"):
                if merged.get(k) is not None and not isinstance(merged.get(k), (int, float)):
                    try:
                        merged[k] = float(str(merged.get(k)).strip())
                    except Exception:
                        merged[k] = None

            if isinstance(merged.get("citation"), dict) and isinstance(merged["citation"].get("year"), (int, float)):
                merged["citation"]["year"] = str(int(merged["citation"]["year"]))

            try:
                review_obj = ReviewOutput.parse_obj(merged)
            except Exception as e:
                console.log(f"[yellow]Pydantic validation failed for {file_name}: {e}. Falling back to sanitized summary.[/yellow]")
                cleaned_summary = _strip_code_fences(review_raw)
                if len(cleaned_summary) > 10000:
                    cleaned_summary = cleaned_summary[:10000] + "..."
                review_obj = ReviewOutput(
                    summary_text=cleaned_summary,
                    strengths=[],
                    weaknesses=[],
                    gaps=[],
                    peer_reviewed=None,
                    critique_score=None,
                    tone=None,
                    sentiment_score=None,
                    semantic_patterns=[],
                    citation={"author": "not reported", "year": "not reported", "suggested_citation": "not reported"},
                )

            review_obj.summary_text = _strip_code_fences(review_obj.summary_text)

            # Console output
            console.rule(f"Review - {file_name}")
            console.print(RichJSON(json.dumps(review_obj.dict(), ensure_ascii=False, indent=2)))
            console.rule()

            # Persist Paper
            try:
                paper = Paper(
                    query_id=project_id,
                    title=metadata.get("title") or (file_name or "Untitled"),
                    abstract=None,
                    doi=None,
                    url=None,
                    publication_year=int(metadata.get("year")) if metadata.get("year") and str(metadata.get("year")).isdigit() else None,
                    journal=None,
                    paper_type=None,
                    citation_count=None,
                    impact_factor=None,
                    fetched_from=None,
                    file_path=stored_path,
                )
                db.add(paper)
                db.commit()
                db.refresh(paper)
            except Exception as db_err:
                db.rollback()
                try:
                    os.remove(stored_path)
                except Exception:
                    pass
                results.append({
                    "file_name": file_name,
                    "paper_id": None,
                    "analysis_id": None,
                    "metadata": metadata,
                    "message": f"❌ DB write (paper) failed: {db_err}",
                })
                continue

            # Persist Analysis
            try:
                analysis = PaperAnalysis(
                    paper_id=paper.paper_id,
                    summary_text=review_obj.summary_text,
                    strengths=_serialize_list_for_db(review_obj.strengths),
                    weaknesses=_serialize_list_for_db(review_obj.weaknesses),
                    gaps=_serialize_list_for_db(review_obj.gaps),
                    peer_reviewed=_to_db_peer_reviewed(review_obj.peer_reviewed),
                    critique_score=review_obj.critique_score,
                    tone=review_obj.tone,
                    sentiment_score=review_obj.sentiment_score,
                    semantic_patterns=_serialize_list_for_db(review_obj.semantic_patterns),
                )
                db.add(analysis)
                db.commit()
                db.refresh(analysis)

                results.append({
                    "file_name": file_name,
                    "paper_id": paper.paper_id,
                    "analysis_id": analysis.analysis_id,
                    "metadata": metadata,
                    "message": "✅ Review generated and stored successfully.",
                })

            except Exception as db_err:
                db.rollback()
                try:
                    os.remove(stored_path)
                except Exception:
                    pass
                results.append({
                    "file_name": file_name,
                    "paper_id": paper.paper_id if paper else None,
                    "analysis_id": None,
                    "metadata": metadata,
                    "message": f"❌ DB write (analysis) failed: {db_err}",
                })

        except Exception as e:
            console.log(f"[red]Failed processing file {file_name}: {e}[/red]")
            results.append({
                "file_name": file_name,
                "paper_id": None,
                "analysis_id": None,
                "metadata": None,
                "message": f"❌ Failed processing file: {e}",
            })

    return JSONResponse(
        content={
            "project_id": project_id,
            "results": results,
            "message": "Completed processing of uploaded files.",
        }
    )


@router.get("/review/{paper_id}")
def get_stored_review(paper_id: int, db: Session = Depends(get_db)):
    analysis = db.query(PaperAnalysis).filter(PaperAnalysis.paper_id == paper_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="No stored literature review found for this paper.")

    out = {
        "paper_id": paper_id,
        "summary_text": analysis.summary_text,
        "strengths": _deserialize_list_from_db(getattr(analysis, "strengths", None)),
        "weaknesses": _deserialize_list_from_db(getattr(analysis, "weaknesses", None)),
        "gaps": _deserialize_list_from_db(getattr(analysis, "gaps", None)),
        "peer_reviewed": True if analysis.peer_reviewed == 1 else (False if analysis.peer_reviewed == 0 else None),
        "critique_score": analysis.critique_score,
        "tone": analysis.tone,
        "sentiment_score": analysis.sentiment_score,
        "semantic_patterns": _deserialize_list_from_db(getattr(analysis, "semantic_patterns", None)),
        "created_at": analysis.created_at,
        "message": "✅ Literature review fetched successfully.",
    }
    return out


@router.get("/project/{project_id}/papers")
def list_project_papers(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found.")

    papers = db.query(Paper).filter(Paper.query_id == project_id).all()
    out = []
    for p in papers:
        analysis = db.query(PaperAnalysis).filter(PaperAnalysis.paper_id == p.paper_id).first()
        file_path = getattr(p, "file_path", None)
        file_ext = None
        if file_path:
            file_ext = os.path.splitext(file_path)[1].lstrip(".").upper() or None
        out.append({
            "paper_id": p.paper_id,
            "title": getattr(p, "title", None),
            "file_type": file_ext,
            "file_path": file_path,
            "has_review": analysis is not None,
        })

    return {
        "project_id": project_id,
        "papers": out,
        "message": "✅ Papers listed successfully.",
    }
@router.delete("/paper/{paper_id}")
def delete_paper(paper_id: int, db: Session = Depends(get_db)):
    """
    Delete a paper and its analysis + stored file (if present).

    Returns JSON with flags indicating what was removed.
    """
    paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found.")

    analysis = db.query(PaperAnalysis).filter(PaperAnalysis.paper_id == paper_id).first()
    stored_path = getattr(paper, "file_path", None)

    analysis_deleted = False
    file_deleted = False

    # Remove DB records inside a transaction
    try:
        if analysis:
            db.delete(analysis)
            analysis_deleted = True

        db.delete(paper)
        db.commit()
    except Exception as e:
        db.rollback()
        console.log(f"[red]DB delete failed for paper {paper_id}: {e}[/red]")
        raise HTTPException(status_code=500, detail=f"DB delete failed: {e}")

    # Attempt to delete file on disk (best-effort)
    if stored_path:
        try:
            if os.path.exists(stored_path):
                os.remove(stored_path)
                file_deleted = True
                # attempt to remove parent dir if empty (optional cleanup)
                parent = os.path.dirname(stored_path)
                try:
                    if parent and os.path.isdir(parent) and not os.listdir(parent):
                        os.rmdir(parent)
                except Exception:
                    # not critical; ignore
                    pass
        except Exception as e:
            console.log(f"[yellow]Failed to remove file {stored_path}: {e}[/yellow]")

    return JSONResponse(
        content={
            "paper_id": paper_id,
            "analysis_deleted": analysis_deleted,
            "file_deleted": file_deleted,
            "message": "✅ Paper deletion attempted (check flags for details).",
        }
    )
