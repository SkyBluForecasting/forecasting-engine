from sqlalchemy.orm import sessionmaker

import forecasting_engine.services.weather_service.main as svc_main
from forecasting_db.models import Asset


def test_weather_service_main_runs_against_db(test_engine, monkeypatch):
    # Seed an asset that needs assignment
    SessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    seed_sess = SessionLocal()
    try:
        seed_sess.add(
            Asset(
                asset_uuid="A1",
                asset_type="distribution_substation",
                name="A1",
                depth=0,
                measured=True,
                latitude=43.691,
                longitude=-79.299,
                weather_site_id=None,
            )
        )
        seed_sess.commit()
    finally:
        seed_sess.close()

    # Patch the service's SessionLocal() to use our test_engine-backed sessions
    monkeypatch.setattr(svc_main, "SessionLocal", SessionLocal)

    # Run the real service main (real IOs + real manager + real DB)
    svc_main.main()

    # Verify it actually updated the DB
    verify_sess = SessionLocal()
    try:
        a1 = verify_sess.query(Asset).filter_by(asset_uuid="A1").one()
        assert a1.weather_site_id is not None
        assert a1.weather_site_id.startswith("cell_")
    finally:
        verify_sess.close()
