from forecasting_db.models import Asset
from .base_io import BaseIO
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd


class AssetsIO(BaseIO):
    """Handles loading asset metadata from the database."""

    def __init__(self, session: Session):
        super().__init__(session)

    def list_assets(self, asset_type: Optional[str] = None) -> List[Asset]:
        """
        Get all assets (optionally filtered by type).

        Args:
            asset_type: e.g. 'substation', 'pv', etc.

        Returns:
            List of Asset ORM objects.
        """
        query = self.session.query(Asset)
        if asset_type:
            query = query.filter(Asset.asset_type == asset_type)
        return query.all()

    def get_asset(self, asset_uuid: str) -> Optional[Asset]:
        """
        Fetch a single asset by UUID.
        """
        return (
            self.session.query(Asset)
            .filter(Asset.asset_uuid == asset_uuid)
            .one_or_none()
        )

    def from_df(self, df: pd.DataFrame, *args, **kwargs):
        """Write assets from DataFrame to DB (not used yet)."""
        raise NotImplementedError("from_df() not implemented for AssetsIO")

    def to_df(self, asset_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load assets into a pandas DataFrame.
        """
        assets = self.list_assets(asset_type)
        if not assets:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "asset_uuid": a.asset_uuid,
                    "name": a.name,
                    "asset_type": a.asset_type,
                    "capacity_kw": a.capacity_kw,
                    "parent_uuid": a.parent_uuid,
                }
                for a in assets
            ]
        )
