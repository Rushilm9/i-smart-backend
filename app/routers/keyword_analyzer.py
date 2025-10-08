from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from langchain.prompts import ChatPromptTemplate

from llm.llm import LLMClient
from core.database import get_db
from model.models import Project

router = APIRouter(prefix="/analyze", tags=["Keyword Analyzer"])

# ------------------------------
# Request / Response Schemas
# ------------------------------
class PromptRequest(BaseModel):
    user_id: int | None = None
    project_name: str | None = None
    project_desc: str | None = None
    prompt: str
    file_type: str | None = None

class KeywordResponse(BaseModel):
    project_id: int
    keywords: list[str]


# ------------------------------
# Initialize LLM
# ------------------------------
llm_client = LLMClient()

# ------------------------------
# Prompt Template
# ------------------------------
prompt_template = ChatPromptTemplate.from_template(
    """
    You are an expert research assistant.

    Task:
    - Given an input research topic or phrase: "{keyword}"
    - Suggest 10–15 highly relevant academic keywords.
    - Keywords should be useful for searching research paper titles in databases (IEEE, ACM, Springer, Scopus, etc.).
    - Include both broad terms and specific related concepts.
    - Prefer technical, domain-specific vocabulary over general words.
    - Do NOT include numbering or explanations, only a clean comma-separated list of keywords.

    Example:
    Input: "machine learning in healthcare"
    Output: machine learning, deep learning, healthcare analytics, medical imaging, clinical decision support, electronic health records, predictive modeling, neural networks, health informatics, patient outcome prediction, diagnostic systems, precision medicine

    Now, generate keywords for: "{keyword}"
    """
)

# ------------------------------
# Endpoint
# ------------------------------
@router.post("/", response_model=KeywordResponse)
async def analyze_keywords(request: PromptRequest, db: Session = Depends(get_db)):
    """
    Generate academic keywords for a topic, save them to the Project table,
    and return the list of extracted keywords.
    """
    # Step 1. Build prompt for LLM
    prompt = prompt_template.format(keyword=request.prompt)

    # Step 2. Get response from LLM
    response_text = llm_client.chat(prompt)
    keywords = [k.strip() for k in response_text.split(",") if k.strip()]

    if not keywords:
        raise HTTPException(status_code=400, detail="No keywords generated.")

    # Step 3. Save to database
    project_entry = Project(
        user_id=request.user_id,
        project_name=request.project_name,
        project_desc=request.project_desc,
        raw_query=request.prompt,
        expanded_query=", ".join(keywords),  # store as comma-separated list
        file_type=request.file_type,
    )

    db.add(project_entry)
    db.commit()
    db.refresh(project_entry)

    # Step 4. Return structured response
    return KeywordResponse(
        project_id=project_entry.project_id,
        keywords=keywords,
    )
