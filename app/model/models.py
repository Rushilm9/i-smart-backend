from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, BigInteger,ForeignKey
from core.database import Base
from sqlalchemy.sql import func

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    password_hash = Column(Text, nullable=False)
    name = Column(String(255))
    email = Column(String(255), unique=True, index=True, nullable=False)
    affiliation = Column(String(255))
    created_at = Column(TIMESTAMP, server_default=func.now())



class Project(Base):
    __tablename__ = "projects"

    project_id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.user_id"), nullable=True, index=True)
    project_name = Column(String(255), nullable=True)
    project_desc = Column(Text, nullable=True)
    raw_query = Column(Text, nullable=True)
    expanded_query = Column(Text, nullable=True)
    file_type = Column(String(50), nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=True, onupdate=func.now())
