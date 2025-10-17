from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from forecasting_db.models import Base
from forecasting_engine.config import DATABASE_URL

# Create synchronous engine
engine = create_engine(DATABASE_URL, echo=False, future=True)

# Create session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Create tables if not exist
Base.metadata.create_all(bind=engine)
