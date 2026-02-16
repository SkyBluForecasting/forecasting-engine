# """Integration tests for weather service with DB.

# Tests weather site assignment, upsertion, and asset weather_site_id updates.
# """

# from __future__ import annotations

# import pytest

# from forecasting_db.models import Asset, WeatherSite


# # ----------------------------
# # Fixtures
# # ----------------------------


# @pytest.fixture
# def asset_with_coords(test_session) -> Asset:
#     """Asset with latitude and longitude coordinates."""
#     pass


# @pytest.fixture
# def multiple_assets_same_cell(test_session) -> list[Asset]:
#     """Multiple assets in the same weather grid cell."""
#     pass


# @pytest.fixture
# def assets_different_cells(test_session) -> list[Asset]:
#     """Assets in different weather grid cells."""
#     pass


# # ----------------------------
# # Weather Site Assignment Integration Tests
# # ----------------------------

# def test_assign_weather_sites_from_db_creates_weather_site(
#     test_session,
#     asset_with_coords,
# ):
#     """Test that assign_weather_sites_from_db creates a new WeatherSite entry in DB."""
#     pass


# def test_assign_weather_sites_from_db_updates_asset_weather_site_id(
#     test_session,
#     asset_with_coords,
# ):
#     """Test that asset.weather_site_id is persisted to DB after assignment."""
#     pass


# def test_assign_weather_sites_from_db_reuses_existing_weather_site(
#     test_session,
#     multiple_assets_same_cell,
# ):
#     """Test that multiple assets in same cell share the same weather_site_id in DB."""
#     pass


# def test_assign_weather_sites_from_db_creates_multiple_sites(
#     test_session,
#     assets_different_cells,
# ):
#     """Test that assets in different cells get different weather_site_ids persisted to DB."""
#     pass
