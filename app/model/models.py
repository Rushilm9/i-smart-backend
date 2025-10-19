# app/model/models.py
from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, Float, TIMESTAMP,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from core.database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(BigInteger, primary_key=True, autoincrement=True, index=True)
    password_hash = Column(Text, nullable=False)
    name = Column(String(255))
    email = Column(String(255), unique=True, index=True, nullable=True)
    affiliation = Column(String(255))
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

    projects = relationship("Project", back_populates="user")


class Project(Base):
    __tablename__ = "projects"

    project_id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.user_id", ondelete="SET NULL"), index=True, nullable=True)
    project_name = Column(String(255), nullable=True)
    project_desc = Column(Text, nullable=True)
    raw_query = Column(Text, nullable=True)
    expanded_query = Column(Text, nullable=True)
    file_type = Column(String(50), nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=True, onupdate=func.now())

    user = relationship("User", back_populates="projects")
    papers = relationship("Paper", back_populates="project")
    recommendations = relationship("Recommendation", back_populates="project")


class Paper(Base):
    __tablename__ = "papers"

    paper_id = Column(BigInteger, primary_key=True, autoincrement=True)
    query_id = Column(BigInteger, ForeignKey("projects.project_id", ondelete="SET NULL"), index=True, nullable=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=True)
    doi = Column(String(100), unique=True, index=True, nullable=True)
    url = Column(Text, nullable=True)
    publication_year = Column(Integer, nullable=True)
    journal = Column(String(255), nullable=True)
    paper_type = Column(String(50), nullable=True)
    citation_count = Column(Integer, nullable=True)
    impact_factor = Column(Float, nullable=True)
    fetched_from = Column(String(100), nullable=True)

    # store uploaded file location only
    file_path = Column(Text, nullable=True)

    ingestion_date = Column(TIMESTAMP, nullable=False, server_default=func.now())

    project = relationship("Project", back_populates="papers")
    authors = relationship("PaperAuthor", back_populates="paper", cascade="all, delete-orphan")
    classifications = relationship("Classification", back_populates="paper", cascade="all, delete-orphan")
    analyses = relationship("PaperAnalysis", back_populates="paper", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="paper")


class Author(Base):
    __tablename__ = "authors"

    author_id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255))
    affiliation = Column(String(255))
    h_index = Column(Integer)
    citation_count = Column(Integer)
    profile_url = Column(Text)

    papers = relationship("PaperAuthor", back_populates="author", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("name", "affiliation", name="uq_author_name_affil"),
    )


class PaperAuthor(Base):
    __tablename__ = "paper_authors"

    paper_id = Column(BigInteger, ForeignKey("papers.paper_id", ondelete="CASCADE"), primary_key=True)
    author_id = Column(BigInteger, ForeignKey("authors.author_id", ondelete="CASCADE"), primary_key=True)

    paper = relationship("Paper", back_populates="authors")
    author = relationship("Author", back_populates="papers")


class Classification(Base):
    __tablename__ = "classifications"

    classification_id = Column(BigInteger, primary_key=True, autoincrement=True)
    paper_id = Column(BigInteger, ForeignKey("papers.paper_id", ondelete="CASCADE"), index=True, nullable=False)
    category_type = Column(String(50))
    category_value = Column(String(255))

    paper = relationship("Paper", back_populates="classifications")


class PaperAnalysis(Base):
    __tablename__ = "paper_analysis"

    analysis_id = Column(BigInteger, primary_key=True, autoincrement=True)
    paper_id = Column(BigInteger, ForeignKey("papers.paper_id", ondelete="CASCADE"), index=True, nullable=False)
    summary_text = Column(Text)
    strengths = Column(Text)
    weaknesses = Column(Text)
    gaps = Column(Text)
    peer_reviewed = Column(Integer)
    critique_score = Column(Float)
    tone = Column(String(50))
    sentiment_score = Column(Float)
    semantic_patterns = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())

    paper = relationship("Paper", back_populates="analyses")


class Recommendation(Base):
    __tablename__ = "recommendations"

    recommendation_id = Column(BigInteger, primary_key=True, autoincrement=True)
    query_id = Column(BigInteger, ForeignKey("projects.project_id", ondelete="CASCADE"), index=True, nullable=True)
    paper_id = Column(BigInteger, ForeignKey("papers.paper_id", ondelete="CASCADE"), index=True, nullable=True)
    relevance_score = Column(Float)
    novelty_score = Column(Float)
    overall_rank = Column(Integer)
    recommended_reason = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("query_id", "paper_id", name="uq_recommendation_query_paper"),
    )

    project = relationship("Project", back_populates="recommendations")
    paper = relationship("Paper", back_populates="recommendations")
