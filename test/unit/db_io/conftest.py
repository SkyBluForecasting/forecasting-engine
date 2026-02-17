import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from forecasting_db.models import Base
from forecasting_engine.db_io.constraint_io import ConstraintsIO
from forecasting_engine.db_io.forecast_io import ForecastIO
from forecasting_engine.db_io.assets_io import AssetsIO
from forecasting_db.models import Asset


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


@pytest.fixture
def assets_io(in_memory_session):
    return AssetsIO(in_memory_session)


@pytest.fixture
def asset_factory(in_memory_session):
    """
    Minimal factory to reduce Asset(...) boilerplate.

    Usage:
        a1 = asset_factory(asset_uuid="A1", asset_type="system")
        asset_factory(asset_uuid="A2", asset_type="system", parent_uuid="A1", depth=1)
    """

    def _make(**overrides):
        defaults = dict(
            asset_uuid="A",
            asset_type="system",
            name="Asset",
            depth=0,
            measured=True,
            parent_uuid=None,
            capacity_kw=None,
            latitude=None,
            longitude=None,
            weather_site_id=None,
        )
        defaults.update(overrides)
        a = Asset(**defaults)
        in_memory_session.add(a)
        in_memory_session.flush()
        return a

    return _make
