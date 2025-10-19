import os
import json
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from core.database import get_db
from model.models import Project
from llm.llm import LLMClient

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

# ====================================================
# Router Initialization
# ====================================================
router = APIRouter(prefix="/keyword", tags=["Keyword Analyzer"])

# Initialize LLM client
llm_client = LLMClient()

# ====================================================
# Response Models
# ====================================================
class FileResult(BaseModel):
    file_name: str
    file_path: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str]

class KeywordResponse(BaseModel):
    project_id: int
    user_id: int
    results: List[FileResult]
    message: str


# ====================================================
# Prompts
# ====================================================
def build_summarizer_prompt(content: str) -> str:
    return f"""
You are an expert academic summarizer.
Summarize the following research paper content clearly and comprehensively in 5 to 7 sentences maximum.

Focus on:
- The main research problem or goal
- The approach or methodology used
- The key findings or results
- Any datasets, experiments, or challenges discussed
- Broader implications or contributions

Make sure the summary is concise (no more than 7 sentences) and suitable for quick academic review.

Content:
{content}
""".strip()


def build_keyword_prompt(content: str) -> str:
    return f"""
You are an expert research analyst.

Analyze the following research content, which may include a paper summary and/or a user-provided topic:

{content}

Your task:
- Extract ONLY relevant, high-quality academic research keywords that directly reflect the core ideas, methods, or applications mentioned.
- Focus on terms that actually appear in or are strongly implied by the content.
- If fewer than 10 keywords are truly relevant, list only those.
- DO NOT invent unrelated or generic research terms.
- Avoid duplicates or overly similar words (e.g., “AI” and “Artificial Intelligence” — keep only one).

⚠️ Output rules:
- Only list the keywords
- One keyword per line
- No numbering, no JSON, no commentary, no quotes
""".strip()


# ====================================================
# Helper: Sentence limiter
# ====================================================
def limit_sentences(text: str, max_sentences: int = 7) -> str:
    """Ensure summary doesn't exceed max_sentences."""
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    truncated = ". ".join(sentences[:max_sentences])
    if truncated and not truncated.endswith("."):
        truncated += "."
    return truncated


# ====================================================
# Endpoint 1️⃣: Analyze & Store Keywords
# ====================================================
@router.post("/analyze", response_model=KeywordResponse)
async def analyze_and_store(
    user_id: int = Form(...),
    project_id: int = Form(...),
    prompt: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    db: Session = Depends(get_db),
):
    """
    Analyze uploaded files and/or text prompts to extract academic keywords.
    Stores the summarized results and keywords into the Project database table.
    """

    # Validate project ownership
    project = (
        db.query(Project)
        .filter(Project.project_id == project_id, Project.user_id == user_id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found for this user")

    if not prompt and not files:
        raise HTTPException(status_code=400, detail="Provide either text or at least one file for analysis.")

    # Base folder setup
    base_folder = os.path.join("uploads", str(user_id), str(project_id))
    os.makedirs(base_folder, exist_ok=True)

    results: List[FileResult] = []
    all_keywords: List[str] = []
    summaries: List[str] = []

    def rel(p: str) -> str:
        try:
            return os.path.relpath(p, start=os.getcwd())
        except Exception:
            return p

    # ====================================================
    # 1️⃣ Process Uploaded Files
    # ====================================================
    if files:
        for file in files:
            file_path = os.path.join(base_folder, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # Extract content
            loader = DoclingLoader(file_path=file_path, export_type=ExportType.MARKDOWN)
            docs = loader.load()
            file_content = "\n\n".join(d.page_content for d in docs)

            # ---- Step 1: Summarize ----
            summary_text = llm_client.chat(build_summarizer_prompt(file_content))
            summary_text = limit_sentences(summary_text, 7)
            summaries.append(f"### {file.filename}\n{summary_text}")

            # ---- Step 2: Combine with user prompt ----
            combined_content = summary_text
            if prompt:
                combined_content += f"\n\nUser provided topic:\n{prompt}"

            # ---- Step 3: Extract Keywords ----
            keyword_output = llm_client.chat(build_keyword_prompt(combined_content))
            keywords = [k.strip() for k in keyword_output.split("\n") if k.strip()]

            # ---- Step 4: Deduplicate ----
            seen = set()
            deduped = []
            for k in keywords:
                key = k.lower()
                if key not in seen:
                    seen.add(key)
                    deduped.append(k)

            all_keywords.extend(deduped)

            # ---- Step 5: Append File Result ----
            results.append(
                FileResult(
                    file_name=file.filename,
                    file_path=rel(file_path),
                    summary=summary_text,
                    keywords=deduped,
                )
            )

    # ====================================================
    # 2️⃣ Only Text Prompt (no files)
    # ====================================================
    if not files and prompt:
        keyword_output = llm_client.chat(build_keyword_prompt(prompt))
        keywords = [k.strip() for k in keyword_output.split("\n") if k.strip()]

        seen = set()
        deduped = []
        for k in keywords:
            key = k.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(k)

        all_keywords.extend(deduped)
        results.append(
            FileResult(
                file_name="text_prompt",
                file_path=None,
                summary=None,
                keywords=deduped,
            )
        )

    # ====================================================
    # 3️⃣ Store to Database
    # ====================================================
    combined_summary = "\n\n".join(summaries) if summaries else None
    relative_paths = [r.file_path for r in results if r.file_path]

    project.file_type = "pdf" if files else "text"
    project.raw_query = prompt or project.raw_query

    expanded_data = {
        "files": relative_paths,
        "keywords": all_keywords,
        "summaries": combined_summary,
    }

    project.expanded_query = json.dumps(expanded_data, ensure_ascii=False, indent=2)

    db.commit()
    db.refresh(project)

    return KeywordResponse(
        project_id=project.project_id,
        user_id=user_id,
        results=results,
        message="✅ Analysis complete. Data stored in DB successfully.",
    )


# ====================================================
# Updated Models for Fetch API
# ====================================================
class ProjectInfo(BaseModel):
    project_id: int
    project_name: Optional[str] = None
    project_desc: Optional[str] = None
    raw_query: Optional[str] = None
    file_type: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    user_id: Optional[int] = None


class KeywordFileInfo(BaseModel):
    file_name: str
    file_path: str


class EnhancedKeywordFetchResponse(BaseModel):
    project: ProjectInfo
    files: List[KeywordFileInfo]
    keywords: List[str]
    summaries: Optional[str] = None
    message: str


# ====================================================
# ✅ Endpoint 2️⃣: Fetch Keywords, Summaries, File Paths, and Project Info
# ====================================================
@router.get("/fetch/{project_id}", response_model=EnhancedKeywordFetchResponse)
def fetch_keywords_and_files(project_id: int, db: Session = Depends(get_db)):
    """
    Fetch previously analyzed keywords, summaries, file paths, and project metadata including raw_query.
    """
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.expanded_query:
        raise HTTPException(status_code=404, detail="No keyword or file data found for this project.")

    try:
        expanded_data = json.loads(project.expanded_query)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Corrupted expanded_query data")

    file_paths = expanded_data.get("files", []) or []
    keywords = expanded_data.get("keywords", []) or []
    summaries = expanded_data.get("summaries", None)

    # Deduplicate keywords
    seen = set()
    deduped_keywords = []
    for k in keywords:
        key = k.lower()
        if key not in seen:
            seen.add(key)
            deduped_keywords.append(k)

    # Build file info
    files_info = []
    for path in file_paths:
        file_name = os.path.basename(path)
        files_info.append(KeywordFileInfo(file_name=file_name, file_path=path))

    # Build project info
    project_info = ProjectInfo(
        project_id=project.project_id,
        project_name=project.project_name,
        project_desc=project.project_desc,
        raw_query=project.raw_query,
        file_type=project.file_type,
        created_at=str(project.created_at) if project.created_at else None,
        updated_at=str(project.updated_at) if project.updated_at else None,
        user_id=project.user_id,
    )

    return EnhancedKeywordFetchResponse(
        project=project_info,
        files=files_info,
        keywords=deduped_keywords,
        summaries=summaries,
        message="✅ Project data (with raw query, file paths, and metadata) fetched successfully."
    )


# ====================================================
# Endpoint 3️⃣: List Uploaded Files by Project ID
# ====================================================
@router.get("/files/{project_id}")
def list_uploaded_files(project_id: int, db: Session = Depends(get_db)):
    """
    Return all uploaded file names and paths for a given project.
    """
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    base_folder = os.path.join("uploads", str(project.user_id), str(project_id))
    if not os.path.exists(base_folder):
        raise HTTPException(status_code=404, detail="No upload folder found for this project")

    files = []
    for f in os.listdir(base_folder):
        full_path = os.path.join(base_folder, f)
        if os.path.isfile(full_path):
            files.append({"file_name": f, "file_path": full_path})

    if not files:
        raise HTTPException(status_code=404, detail="No files found for this project")

    return {
        "project_id": project_id,
        "files": files,
        "message": "✅ Files fetched successfully."
    }
