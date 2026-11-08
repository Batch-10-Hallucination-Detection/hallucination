"""
Database session management and schema initialization.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from backend.database.models import Base, Domain, ModelInfo

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./hallucination_detection.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 15} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initializes tables and seeds initial domain/model records."""
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Seed Domains
        if not db.query(Domain).filter_by(domain_id="healthcare").first():
            db.add(Domain(
                domain_id="healthcare",
                domain_name="Healthcare & Clinical Medicine",
                description="High-stakes clinical QA, drug dosages, contraindications, and clinical guidelines."
            ))
        if not db.query(Domain).filter_by(domain_id="software").first():
            db.add(Domain(
                domain_id="software",
                domain_name="Software Development",
                description="Programming Q&A, syntax, library documentation, and API specifications."
            ))

        # Seed Models
        default_models = [
            ("llama-3-8b", "Meta-Llama-3-8B-Instruct", "3.1"),
            ("mistral-7b", "Mistral-7B-Instruct-v0.3", "0.3"),
            ("qwen-2.5-7b", "Qwen-2.5-7B-Instruct", "2.5")
        ]
        for m_id, m_name, m_ver in default_models:
            if not db.query(ModelInfo).filter_by(model_id=m_id).first():
                db.add(ModelInfo(model_id=m_id, model_name=m_name, model_version=m_ver))

        db.commit()
    finally:
        db.close()


def get_db():
    """Dependency generator for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

