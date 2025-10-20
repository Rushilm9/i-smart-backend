# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.database import Base, engine
from routers import auth, keyword_analyzer, projects,research,fetch_papers,literature_review

# --- Create tables (optional in dev mode) ---
Base.metadata.create_all(bind=engine)

# --- Initialize FastAPI ---
app = FastAPI(
    title="Research AI API",
    version="1.0",
    description="Backend API for Research AI platform"
)

# --- CORS Middleware ---
# This allows requests from any frontend (e.g., React, Vue, etc.)
# You can later restrict `allow_origins` to your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],          # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],          # Allow all headers
)

# --- Include Routers ---
app.include_router(auth.router)
app.include_router(keyword_analyzer.router)
app.include_router(projects.router)
app.include_router(research.router)
app.include_router(fetch_papers.router)
app.include_router(literature_review.router)

# --- Root Endpoint ---
@app.get("/")
def root():
    return {"message": "🚀 API is running!"}
