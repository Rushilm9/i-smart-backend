# app/routers/papers.py
"""
Paper Retrieval & Recommendation Endpoints
------------------------------------------
Provides ORM-based APIs for:
- Listing all papers under a project
- Fetching full details of a single paper
- Getting recommended papers for a project
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from core.database import get_db
from model.models import Project, Paper, Recommendation, PaperAnalysis
from rich.console import Console

console = Console(log_path=False, emoji=True)

# Router setup
router = APIRouter(prefix="/papers", tags=["Papers"])

# =====================================================================
# API 1: Fetch Papers for a Project
# =====================================================================

@router.get("/project/{project_id}")
def get_papers_for_project(
    project_id: int,
    limit: int = Query(100, ge=1, le=500, description="Maximum number of papers to fetch"),
    db: Session = Depends(get_db),
):
    """
    Fetch all papers associated with a given project (via project_id).
    Returns basic paper details (title, DOI, year, etc.).
    """
    console.rule(f":page_facing_up: [bold]Fetch Papers[/] • Project ID: [cyan]{project_id}[/]")

    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")

    query = (
        db.query(Paper, Recommendation)
        .join(Recommendation, Recommendation.paper_id == Paper.paper_id, isouter=True)
        .filter(Paper.query_id == project_id)
        .limit(limit)
    )

    records = query.all()
    if not records:
        raise HTTPException(status_code=404, detail=f"No papers found for project ID {project_id}")

    papers_out = []
    for paper, rec in records:
        papers_out.append({
            "paper_id": paper.paper_id,
            "title": paper.title,
            "doi": paper.doi,
            "abstract": paper.abstract,
            "journal": paper.journal,
            "publication_year": paper.publication_year,
            "citation_count": paper.citation_count,
            "impact_factor": paper.impact_factor,
            "sjr_quartile": getattr(paper, "sjr_quartile", None),
            "h_index": getattr(paper, "h_index", None),
            "is_open_access": getattr(paper, "is_open_access", None),
            "oa_url": getattr(paper, "oa_url", None),
            "relevance_score": rec.relevance_score if rec else None,
            "overall_rank": rec.overall_rank if rec else None,
        })

    console.log(f"[green]Returned {len(papers_out)} papers for project {project_id}[/]")

    return {
        "project_id": project_id,
        "project_name": getattr(project, "name", None),
        "paper_count": len(papers_out),
        "papers": papers_out,
    }


# =====================================================================
# API 2: Fetch Full Paper Details (ORM only)
# =====================================================================

@router.get("/detail/{paper_id}")
def get_paper_details(
    paper_id: int,
    db: Session = Depends(get_db),
):
    """
    Fetch complete details for a single paper using ORM only:
    - Paper metadata
    - Linked authors
    - Classifications
    - Paper analysis (latest)
    - Recommendation info
    """
    console.rule(f":bookmark_tabs: [bold]Fetch Paper Details[/] • Paper ID: [cyan]{paper_id}[/]")

    paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found")

    # --- Authors (via PaperAuthor → Author) ---
    authors_out = []
    for pa in paper.authors:
        author = pa.author
        if author:
            authors_out.append({
                "author_id": author.author_id,
                "name": author.name,
                "affiliation": author.affiliation,
                "h_index": author.h_index,
                "citation_count": author.citation_count,
                "profile_url": author.profile_url,
            })

    # --- Classifications ---
    classifications_out = [
        {
            "classification_id": c.classification_id,
            "category_type": c.category_type,
            "category_value": c.category_value,
        }
        for c in getattr(paper, "classifications", [])
    ]

    # --- Latest Paper Analysis ---
    analysis_obj = (
        db.query(PaperAnalysis)
        .filter(PaperAnalysis.paper_id == paper_id)
        .order_by(PaperAnalysis.created_at.desc())
        .first()
    )
    analysis_out = None
    if analysis_obj:
        analysis_out = {
            "analysis_id": analysis_obj.analysis_id,
            "summary_text": analysis_obj.summary_text,
            "strengths": analysis_obj.strengths,
            "weaknesses": analysis_obj.weaknesses,
            "gaps": analysis_obj.gaps,
            "peer_reviewed": analysis_obj.peer_reviewed,
            "critique_score": analysis_obj.critique_score,
            "tone": analysis_obj.tone,
            "sentiment_score": analysis_obj.sentiment_score,
            "semantic_patterns": analysis_obj.semantic_patterns,
            "created_at": analysis_obj.created_at,
        }

    # --- Recommendation (if any) ---
    recommendation = (
        db.query(Recommendation)
        .filter(Recommendation.paper_id == paper_id)
        .first()
    )
    rec_out = None
    if recommendation:
        rec_out = {
            "recommendation_id": recommendation.recommendation_id,
            "project_id": recommendation.query_id,
            "relevance_score": recommendation.relevance_score,
            "novelty_score": recommendation.novelty_score,
            "overall_rank": recommendation.overall_rank,
            "recommended_reason": recommendation.recommended_reason,
            "created_at": recommendation.created_at,
        }

    # --- Final response ---
    result = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "doi": paper.doi,
        "abstract": paper.abstract,
        "journal": paper.journal,
        "publication_year": paper.publication_year,
        "citation_count": paper.citation_count,
        "impact_factor": paper.impact_factor,
        "sjr_quartile": getattr(paper, "sjr_quartile", None),
        "h_index": getattr(paper, "h_index", None),
        "is_open_access": getattr(paper, "is_open_access", None),
        "oa_url": getattr(paper, "oa_url", None),
        "url": paper.url,
        "fetched_from": paper.fetched_from,
        "ingestion_date": paper.ingestion_date,
        "authors": authors_out,
        "classifications": classifications_out,
        "analysis": analysis_out,
        "recommendation": rec_out,
    }

    console.log(f"[green]Returned ORM details for paper {paper_id}[/]")
    return result


# =====================================================================
# API 3: Get Recommended Papers for a Project
# =====================================================================
@router.get("/recommended/{project_id}")
def get_recommended_papers_for_project(
    project_id: int,
    limit: int = Query(
        200,
        ge=1,
        le=1000,
        description="Maximum number of recommended papers to fetch",
    ),
    db: Session = Depends(get_db),
):
    """
    Fetch all recommended papers for a given project using ORM.
    Includes paper details + recommendation scores (relevance, novelty, rank).
    """
    console.rule(
        f":star2: [bold]Fetch Recommended Papers[/] • Project ID: [cyan]{project_id}[/]"
    )

    # --- Validate project existence ---
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=404, detail=f"Project with ID {project_id} not found"
        )

    # --- ORM query: Recommendation + Paper join ---
    # TiDB/MySQL don't support NULLS LAST → manual workaround
    if db.bind.dialect.name in ("mysql", "tidb", "sqlite"):
        order_clause = [
            Recommendation.overall_rank.is_(None),
            Recommendation.overall_rank.asc(),
        ]
    else:
        # For PostgreSQL / Oracle (supports NULLS LAST)
        order_clause = [Recommendation.overall_rank.asc().nullslast()]

    recommendations = (
        db.query(Recommendation)
        .join(Paper, Paper.paper_id == Recommendation.paper_id)
        .filter(Recommendation.query_id == project_id)
        .order_by(*order_clause)
        .limit(limit)
        .all()
    )

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail=f"No recommended papers found for project ID {project_id}",
        )

    # --- Format output ---
    results = []
    for rec in recommendations:
        paper = rec.paper
        results.append(
            {
                "recommendation_id": rec.recommendation_id,
                "paper_id": paper.paper_id,
                "title": paper.title,
                "doi": paper.doi,
                "abstract": paper.abstract,
                "journal": paper.journal,
                "publication_year": paper.publication_year,
                "citation_count": paper.citation_count,
                "impact_factor": paper.impact_factor,
                "sjr_quartile": getattr(paper, "sjr_quartile", None),
                "h_index": getattr(paper, "h_index", None),
                "is_open_access": getattr(paper, "is_open_access", None),
                "oa_url": getattr(paper, "oa_url", None),
                "relevance_score": rec.relevance_score,
                "novelty_score": rec.novelty_score,
                "overall_rank": rec.overall_rank,
                "recommended_reason": rec.recommended_reason,
                "created_at": rec.created_at,
            }
        )

    console.log(
        f"[green]Returned {len(results)} recommended papers for project {project_id}[/]"
    )

    return {
        "project_id": project.project_id,
        "project_name": getattr(project, "name", None)
        or getattr(project, "project_name", None),
        "recommendation_count": len(results),
        "recommended_papers": results,
    }