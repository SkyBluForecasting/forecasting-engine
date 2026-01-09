import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from forecasting_db.models import Base
from forecasting_engine.db_io.constraint_io import ConstraintsIO
from forecasting_engine.db_io.forecast_io import ForecastIO


@pytest.fixture
def in_memory_session():
    engine = create_engine("sqlite:///:memory:", future=True)

    # SQLite doesn't support schemas
    for table in Base.metadata.tables.values():
        table.schema = None

    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine, future=True)
    session = Session()

    try:
        yield session
    finally:
        session.close()


# # ----------------------------
# # Sample test data fixtures
# # ----------------------------


@pytest.fixture
def constraints_io(in_memory_session):
    return ConstraintsIO(in_memory_session)


@pytest.fixture
def forecast_io(in_memory_session):
    return ForecastIO(in_memory_session)
