import os
from pathlib import Path
from dotenv import load_dotenv

# env_path = Path(__file__).resolve().parents[1] / ".env"
# load_dotenv(dotenv_path=env_path)
load_dotenv(override=True)
class Settings:
    TIDB_USER: str = os.getenv("TIDB_USER")
    TIDB_PASSWORD: str = os.getenv("TIDB_PASSWORD")
    TIDB_HOST: str = os.getenv("TIDB_HOST")
    TIDB_PORT: str = os.getenv("TIDB_PORT", "4000")
    TIDB_DATABASE: str = os.getenv("TIDB_DATABASE")

    CORS_ALLOW_ORIGINS: str = os.getenv("CORS_ALLOW_ORIGINS", "*")
    CORS_ALLOW_CREDENTIALS: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "True") == "True"
    CORS_ALLOW_METHODS: str = os.getenv("CORS_ALLOW_METHODS", "*")
    CORS_ALLOW_HEADERS: str = os.getenv("CORS_ALLOW_HEADERS", "*")

settings = Settings()
