from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from urllib.parse import quote_plus
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# --- Database Config from .env ---
TIDB_USER = os.getenv("TIDB_USER")
TIDB_PASSWORD = os.getenv("TIDB_PASSWORD")
TIDB_HOST = os.getenv("TIDB_HOST")
TIDB_PORT = os.getenv("TIDB_PORT", "4000")
TIDB_DATABASE = os.getenv("TIDB_DATABASE")

# --- Validation ---
if not all([TIDB_USER, TIDB_PASSWORD, TIDB_HOST, TIDB_DATABASE]):
    raise RuntimeError("❌ Missing database environment variables. Check your .env file.")

# --- Safely encode credentials ---
user = quote_plus(TIDB_USER)
pwd = quote_plus(TIDB_PASSWORD)

# --- Build Database URL ---
DATABASE_URL = f"mysql+mysqlconnector://{user}:{pwd}@{TIDB_HOST}:{TIDB_PORT}/{TIDB_DATABASE}"

# --- SQLAlchemy Engine ---
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # avoid stale connections
    pool_recycle=3600,    # reconnect every hour
)

# --- Session Factory ---
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Base class for models ---
Base = declarative_base()

# --- Dependency for FastAPI ---
def get_db():
    """Provide a database session for request scope."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
