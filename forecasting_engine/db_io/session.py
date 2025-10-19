from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from forecasting_db.models import Base
from forecasting_engine.config import DATABASE_URL

# Create synchronous engine
engine = create_engine(DATABASE_URL, echo=False, future=True)

# If using Postgres, make sure the schema exists
if DATABASE_URL.startswith("postgres"):
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS openstef"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS product"))
        conn.commit()

# Create session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Create tables if not exist
Base.metadata.create_all(bind=engine)
