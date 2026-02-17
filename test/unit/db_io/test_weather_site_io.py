import pytest
import pandas as pd

from forecasting_db.models import WeatherSite
from forecasting_engine.db_io.weather_site_io import WeatherSitesIO


@pytest.fixture
def weather_sites_io(in_memory_session):
    return WeatherSitesIO(in_memory_session)


@pytest.fixture
def site_factory(in_memory_session):
    """
    Minimal factory to reduce boilerplate.

    Usage:
      site_factory("SITE1", 43.7, -79.4)
    """

    def _make(weather_site_id="SITE1", site_lat=1.0, site_long=2.0):
        s = WeatherSite(
            weather_site_id=weather_site_id,
            site_lat=site_lat,
            site_long=site_long,
        )
        in_memory_session.add(s)
        in_memory_session.flush()  # don't commit unless the test needs it
        return s

    return _make


def ids(rows):
    return [r.weather_site_id for r in rows]


# ----------------------------
# list_sites
# ----------------------------


def test_list_sites_returns_all(weather_sites_io, site_factory):
    site_factory("S1", 10.0, 20.0)
    site_factory("S2", 11.0, 21.0)

    result = weather_sites_io.list_sites()
    assert set(ids(result)) == {"S1", "S2"}


def test_list_sites_empty_returns_empty_list(weather_sites_io):
    assert weather_sites_io.list_sites() == []


# ----------------------------
# get_site
# ----------------------------


def test_get_site_found(weather_sites_io, site_factory):
    site_factory("S1", 10.0, 20.0)

    s = weather_sites_io.get_site("S1")
    assert s is not None
    assert s.weather_site_id == "S1"
    assert s.site_lat == 10.0
    assert s.site_long == 20.0


def test_get_site_not_found_returns_none(weather_sites_io):
    assert weather_sites_io.get_site("DOES_NOT_EXIST") is None


# ----------------------------
# exists
# ----------------------------


def test_exists_true(weather_sites_io, site_factory):
    site_factory("S1", 10.0, 20.0)
    assert weather_sites_io.exists("S1") is True


def test_exists_false(weather_sites_io):
    assert weather_sites_io.exists("NOPE") is False


# ----------------------------
# upsert_site
# ----------------------------


def test_upsert_site_inserts_and_returns_true(weather_sites_io):
    inserted = weather_sites_io.upsert_site("S1", 10.0, 20.0)
    assert inserted is True

    # verify persisted
    s = weather_sites_io.get_site("S1")
    assert s is not None
    assert s.weather_site_id == "S1"
    assert s.site_lat == 10.0
    assert s.site_long == 20.0


def test_upsert_site_when_exists_returns_false_and_does_not_change_existing(
    weather_sites_io, site_factory
):
    # pre-existing
    site_factory("S1", 10.0, 20.0)

    inserted = weather_sites_io.upsert_site("S1", 99.0, 88.0)
    assert inserted is False

    # verify no update occurred
    s = weather_sites_io.get_site("S1")
    assert s.site_lat == 10.0
    assert s.site_long == 20.0


# ----------------------------
# to_df / from_df
# ----------------------------


def test_to_df_empty_returns_empty_df(weather_sites_io):
    df = weather_sites_io.to_df()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_to_df_returns_dataframe(weather_sites_io, site_factory):
    site_factory("S1", 10.0, 20.0)

    df = weather_sites_io.to_df()

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"weather_site_id", "site_lat", "site_long"}
    assert df.shape[0] == 1
    assert df.loc[0, "weather_site_id"] == "S1"
    assert df.loc[0, "site_lat"] == 10.0
    assert df.loc[0, "site_long"] == 20.0


def test_from_df_not_implemented(weather_sites_io):
    with pytest.raises(NotImplementedError, match="from_df"):
        weather_sites_io.from_df(pd.DataFrame())
