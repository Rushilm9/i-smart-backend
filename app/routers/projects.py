# app/routers/projects.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from core.database import get_db
from model.models import Project


router = APIRouter(prefix="/projects", tags=["Projects"])


# ------------------------------
# Request & Response Schemas
# ------------------------------
class ProjectCreateRequest(BaseModel):
    user_id: int
    project_name: str
    project_desc: Optional[str] = None


class ProjectCreateResponse(BaseModel):
    project_id: int
    project_name: str


class ProjectUpdateRequest(BaseModel):
    project_name: Optional[str] = None
    project_desc: Optional[str] = None
    raw_query: Optional[str] = None
    expanded_query: Optional[str] = None
    file_type: Optional[str] = None


class ProjectDetail(BaseModel):
    project_id: int
    user_id: int
    project_name: str
    project_desc: Optional[str] = None
    raw_query: Optional[str] = None
    expanded_query: Optional[str] = None
    file_type: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
       from_attributes = True


# ------------------------------
# Routes
# ------------------------------
@router.post("/create", response_model=ProjectCreateResponse)
def create_project(request: ProjectCreateRequest, db: Session = Depends(get_db)):
    """
    Create a new project and return project ID + project name.
    """
    try:
        new_project = Project(
            user_id=request.user_id,
            project_name=request.project_name,
            project_desc=request.project_desc,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(new_project)
        db.commit()
        db.refresh(new_project)

        return ProjectCreateResponse(
            project_id=new_project.project_id,
            project_name=new_project.project_name
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating project: {e}"
        )


@router.get("/user/{user_id}", response_model=List[ProjectDetail])
def get_projects_by_user(user_id: int, db: Session = Depends(get_db)):
    """
    Get all project details for a given user ID.
    """
    projects = db.query(Project).filter(Project.user_id == user_id).all()
    if not projects:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No projects found for this user"
        )

    return projects


@router.put("/{project_id}", response_model=ProjectDetail)
def update_project(project_id: int, request: ProjectUpdateRequest, db: Session = Depends(get_db)):
    """
    Update an existing project's details (partial updates allowed).
    """
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    # Apply changes if provided
    if request.project_name is not None:
        project.project_name = request.project_name
    if request.project_desc is not None:
        project.project_desc = request.project_desc
    if request.raw_query is not None:
        project.raw_query = request.raw_query
    if request.expanded_query is not None:
        project.expanded_query = request.expanded_query
    if request.file_type is not None:
        project.file_type = request.file_type

    project.updated_at = datetime.utcnow()

    try:
        db.commit()
        db.refresh(project)
        return project
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating project: {e}"
        )


@router.delete("/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    """
    Delete a project by its ID.
    """
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    try:
        db.delete(project)
        db.commit()
        return {"message": f"✅ Project ID {project_id} deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting project: {e}"
        )
