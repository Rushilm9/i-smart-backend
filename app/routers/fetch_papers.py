# app/routers/papers.py
"""
Paper Retrieval & Recommendation Endpoints
------------------------------------------
Provides ORM-based APIs for:
- Listing all (unanalyzed) papers under a project
- Fetching full details of a single paper
- Getting recommended (unanalyzed) papers for a project
- Project-wise totals (papers vs analyzed papers)
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, distinct
from sqlalchemy.orm import Session
from core.database import get_db
from model.models import Project, Paper, Recommendation, PaperAnalysis
from rich.console import Console

console = Console(log_path=False, emoji=True)

# Router setup
router = APIRouter(prefix="/papers", tags=["Papers"])


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _analyzed_paper_ids_subq(db: Session):
    """
    Subquery that returns paper_ids that have at least one analysis row.
    Used to EXCLUDE analyzed papers from list endpoints.
    """
    return db.query(PaperAnalysis.paper_id).distinct().subquery()


# =====================================================================
# API 1: Fetch (Unanalyzed) Papers for a Project — from papers table only
# =====================================================================
@router.get("/project/{project_id}")
def get_papers_for_project(
    project_id: int,
    limit: int = Query(
        100, ge=1, le=500, description="Maximum number of papers to fetch"
    ),
    db: Session = Depends(get_db),
):
    """
    Fetch papers associated with a given project (via project_id) **from papers table only**.
    Excludes papers that already exist in paper_analysis.
    Returns basic paper details (title, DOI, year, etc.).
    """
    console.rule(
        f":page_facing_up: [bold]Fetch Papers[/] • Project ID: [cyan]{project_id}[/]"
    )

    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=404, detail=f"Project with ID {project_id} not found"
        )

    analyzed_subq = _analyzed_paper_ids_subq(db)

    query = (
        db.query(Paper, Recommendation)
        .outerjoin(Recommendation, Recommendation.paper_id == Paper.paper_id)
        .filter(Paper.query_id == project_id)
        .filter(~Paper.paper_id.in_(analyzed_subq))  # exclude analyzed papers
        .limit(limit)
    )

    records = query.all()
    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No (unanalyzed) papers found for project ID {project_id}",
        )

    papers_out = []
    for paper, rec in records:
        papers_out.append(
            {
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
            }
        )

    console.log(
        f"[green]Returned {len(papers_out)} (unanalyzed) papers for project {project_id}[/]"
    )

    return {
        "project_id": project_id,
        "project_name": getattr(project, "project_name", None)
        or getattr(project, "name", None),
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
    Fetch complete details for a single paper using ORM:
    - Paper metadata (from papers)
    - Linked authors
    - Classifications
    - Latest Paper analysis (if any) — informational only
    - Recommendation info (if any)
    NOTE: Listing endpoints never pull from paper_analysis. This detail endpoint
    can still show analysis if it exists (for context).
    """
    console.rule(
        f":bookmark_tabs: [bold]Fetch Paper Details[/] • Paper ID: [cyan]{paper_id}[/]"
    )

    paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
    if not paper:
        raise HTTPException(
            status_code=404, detail=f"Paper with ID {paper_id} not found"
        )

    # Authors (via PaperAuthor → Author)
    authors_out = []
    for pa in getattr(paper, "authors", []):
        author = getattr(pa, "author", None)
        if author:
            authors_out.append(
                {
                    "author_id": author.author_id,
                    "name": author.name,
                    "affiliation": author.affiliation,
                    "h_index": author.h_index,
                    "citation_count": author.citation_count,
                    "profile_url": author.profile_url,
                }
            )

    # Classifications
    classifications_out = [
        {
            "classification_id": c.classification_id,
            "category_type": c.category_type,
            "category_value": c.category_value,
        }
        for c in getattr(paper, "classifications", [])
    ]

    # Latest Paper Analysis (if exists)
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

    # Recommendation (if any)
    recommendation = (
        db.query(Recommendation).filter(Recommendation.paper_id == paper_id).first()
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
# API 3: Get (Unanalyzed) Recommended Papers for a Project
# =====================================================================
@router.get("/recommended/{project_id}")
def get_recommended_papers_for_project(
    project_id: int,
    limit: int = Query(
        200, ge=1, le=1000, description="Maximum number of recommended papers to fetch"
    ),
    db: Session = Depends(get_db),
):
    """
    Fetch recommended papers for a given project using ORM.
    Includes paper details + recommendation scores (relevance, novelty, rank).
    EXCLUDES any paper that already has an entry in paper_analysis.
    """
    console.rule(
        f":star2: [bold]Fetch Recommended Papers[/] • Project ID: [cyan]{project_id}[/]"
    )

    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=404, detail=f"Project with ID {project_id} not found"
        )

    # Order clause (NULLS LAST workaround for MySQL/TiDB/SQLite)
    dialect_name = getattr(getattr(db, "bind", None), "dialect", None)
    dialect_name = getattr(dialect_name, "name", None)
    if dialect_name in ("mysql", "tidb", "sqlite"):
        order_clause = [
            Recommendation.overall_rank.is_(None),
            Recommendation.overall_rank.asc(),
        ]
    else:
        order_clause = [Recommendation.overall_rank.asc().nullslast()]

    analyzed_subq = _analyzed_paper_ids_subq(db)

    recommendations = (
        db.query(Recommendation)
        .join(Paper, Paper.paper_id == Recommendation.paper_id)
        .filter(Recommendation.query_id == project_id)
        .filter(~Recommendation.paper_id.in_(analyzed_subq))  # exclude analyzed
        .order_by(*order_clause)
        .limit(limit)
        .all()
    )

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail=f"No (unanalyzed) recommended papers found for project ID {project_id}",
        )

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
        f"[green]Returned {len(results)} (unanalyzed) recommended papers for project {project_id}[/]"
    )

    return {
        "project_id": project.project_id,
        "project_name": getattr(project, "project_name", None)
        or getattr(project, "name", None),
        "recommendation_count": len(results),
        "recommended_papers": results,
    }


# =====================================================================
# API 4: Project-wise Totals (single project)
# =====================================================================
@router.get("/stats/project/{project_id}")
def get_project_totals(
    project_id: int,
    db: Session = Depends(get_db),
):
    """
    Returns counts for a single project:
    - total_papers: number of rows in papers where query_id = project_id
    - analyzed_papers: count of distinct paper_ids present in paper_analysis for that project's papers
    - unanalyzed_papers: total_papers - analyzed_papers
    """
    console.rule(
        f":bar_chart: [bold]Project Totals[/] • Project ID: [cyan]{project_id}[/]"
    )

    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=404, detail=f"Project with ID {project_id} not found"
        )

    total_papers = (
        db.query(func.count(Paper.paper_id))
        .filter(Paper.query_id == project_id)
        .scalar()
        or 0
    )

    analyzed_papers = (
        db.query(func.count(distinct(PaperAnalysis.paper_id)))
        .join(Paper, Paper.paper_id == PaperAnalysis.paper_id)
        .filter(Paper.query_id == project_id)
        .scalar()
        or 0
    )

    result = {
        "project_id": project_id,
        "project_name": getattr(project, "project_name", None)
        or getattr(project, "name", None),
        "total_papers": total_papers,
        "analyzed_papers": analyzed_papers,
        "unanalyzed_papers": max(total_papers - analyzed_papers, 0),
    }

    console.log(f"[green]Totals for project {project_id}: {result}[/]")
    return result


# =====================================================================
# API 5: Project-wise Totals (all projects)
# =====================================================================
@router.get("/stats/projects")
def get_all_projects_totals(
    db: Session = Depends(get_db),
    limit: int = Query(
        1000, ge=1, le=10000, description="Max number of projects to return"
    ),
):
    """
    Returns project-wise totals across all projects:
    - total_papers: count(papers.paper_id) per project
    - analyzed_papers: count(distinct paper_analysis.paper_id) per project
    - unanalyzed_papers: total_papers - analyzed_papers
    """
    console.rule(":bar_chart: [bold]Project-wise Totals[/]")

    # Subquery: total papers per project
    total_papers_sq = (
        db.query(
            Paper.query_id.label("project_id"),
            func.count(Paper.paper_id).label("total_papers"),
        )
        .group_by(Paper.query_id)
        .subquery()
    )

    # Subquery: analyzed papers per project (join via papers to get query_id)
    analyzed_papers_sq = (
        db.query(
            Paper.query_id.label("project_id"),
            func.count(distinct(PaperAnalysis.paper_id)).label("analyzed_papers"),
        )
        .join(PaperAnalysis, PaperAnalysis.paper_id == Paper.paper_id)
        .group_by(Paper.query_id)
        .subquery()
    )

    rows = (
        db.query(
            Project.project_id,
            Project.project_name,
            func.coalesce(total_papers_sq.c.total_papers, 0).label("total_papers"),
            func.coalesce(analyzed_papers_sq.c.analyzed_papers, 0).label(
                "analyzed_papers"
            ),
        )
        .outerjoin(total_papers_sq, total_papers_sq.c.project_id == Project.project_id)
        .outerjoin(
            analyzed_papers_sq, analyzed_papers_sq.c.project_id == Project.project_id
        )
        .order_by(Project.project_id.asc())
        .limit(limit)
        .all()
    )

    if not rows:
        return {"projects": [], "project_count": 0}

    results = []
    for project_id, project_name, total_papers, analyzed_papers in rows:
        total_papers = total_papers or 0
        analyzed_papers = analyzed_papers or 0
        results.append(
            {
                "project_id": project_id,
                "project_name": project_name,
                "total_papers": total_papers,
                "analyzed_papers": analyzed_papers,
                "unanalyzed_papers": max(total_papers - analyzed_papers, 0),
            }
        )

    console.log(f"[green]Returned totals for {len(results)} projects[/]")

    return {
        "projects": results,
        "project_count": len(results),
    }
