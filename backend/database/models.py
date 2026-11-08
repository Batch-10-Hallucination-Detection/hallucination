"""
Database Schema and SQLAlchemy Models.
Fulfills Section 4 Database Design (PostgreSQL / SQLite compatible)
with all 10 core tables and relational mappings.
"""

from datetime import datetime
import uuid
from typing import Optional, List
from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    Integer,
    DateTime,
    ForeignKey,
    create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    user_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("AnalysisSession", back_populates="user", cascade="all, delete-orphan")


class Domain(Base):
    __tablename__ = "domains"
    domain_id = Column(String(50), primary_key=True)
    domain_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)

    sessions = relationship("AnalysisSession", back_populates="domain")


class ModelInfo(Base):
    __tablename__ = "models"
    model_id = Column(String(50), primary_key=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), default="1.0")

    responses = relationship("Response", back_populates="model")
    evaluation_results = relationship("EvaluationResult", back_populates="model")


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"
    session_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=True)
    domain_id = Column(String(50), ForeignKey("domains.domain_id"), nullable=False)
    title = Column(String(200), default="Untitled Analysis")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    domain = relationship("Domain", back_populates="sessions")
    prompts = relationship("Prompt", back_populates="session", cascade="all, delete-orphan")


class Prompt(Base):
    __tablename__ = "prompts"
    prompt_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("analysis_sessions.session_id"), nullable=False)
    prompt_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("AnalysisSession", back_populates="prompts")
    responses = relationship("Response", back_populates="prompt", cascade="all, delete-orphan")


class Response(Base):
    __tablename__ = "responses"
    response_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_id = Column(String(36), ForeignKey("prompts.prompt_id"), nullable=False)
    model_id = Column(String(50), ForeignKey("models.model_id"), nullable=False)
    response_text = Column(Text, nullable=False)
    samples_json = Column(Text, nullable=True)  # JSON-encoded array of sampled responses
    created_at = Column(DateTime, default=datetime.utcnow)

    prompt = relationship("Prompt", back_populates="responses")
    model = relationship("ModelInfo", back_populates="responses")
    claims = relationship("Claim", back_populates="response", cascade="all, delete-orphan")


class Claim(Base):
    __tablename__ = "claims"
    claim_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    response_id = Column(String(36), ForeignKey("responses.response_id"), nullable=False)
    claim_text = Column(Text, nullable=False)
    label = Column(String(50), default="Unverified")  # Supported, Contradicted, Unverifiable

    response = relationship("Response", back_populates="claims")
    detection_result = relationship("DetectionResult", back_populates="claim", uselist=False, cascade="all, delete-orphan")
    evidence_items = relationship("Evidence", back_populates="claim", cascade="all, delete-orphan")


class DetectionResult(Base):
    __tablename__ = "detection_results"
    result_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    claim_id = Column(String(36), ForeignKey("claims.claim_id"), nullable=False, unique=True)
    eigen_score = Column(Float, default=0.0)
    consistency_score = Column(Float, default=0.0)
    entropy_score = Column(Float, default=0.0)
    retrieval_label = Column(String(50), default="Unverifiable")
    final_score = Column(Float, default=0.0)
    risk_level = Column(String(50), default="Moderate Risk")
    explanation = Column(Text, nullable=True)

    claim = relationship("Claim", back_populates="detection_result")


class Evidence(Base):
    __tablename__ = "evidence"
    evidence_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    claim_id = Column(String(36), ForeignKey("claims.claim_id"), nullable=False)
    source = Column(String(200), nullable=False)
    evidence_text = Column(Text, nullable=False)
    relevance_score = Column(Float, default=0.0)

    claim = relationship("Claim", back_populates="evidence_items")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    evaluation_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String(50), ForeignKey("models.model_id"), nullable=False)
    experiment_id = Column(String(20), default="E7")
    accuracy = Column(Float, default=0.0)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    f1_score = Column(Float, default=0.0)
    auroc = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("ModelInfo", back_populates="evaluation_results")

