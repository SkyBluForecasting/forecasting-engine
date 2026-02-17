from forecasting_db.models import Asset
from .base_io import BaseIO
from sqlalchemy.orm import Session, aliased
from sqlalchemy import func
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

    def list_leaf_assets(self, asset_type: Optional[str] = None) -> List[Asset]:
        """
        Return only leaf-node assets (assets with no children).
        """
        Child = aliased(Asset)

        query = (
            self.session.query(Asset)
            .outerjoin(Child, Child.parent_uuid == Asset.asset_uuid)
            .filter(Child.asset_uuid.is_(None))
        )

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

    def list_non_leaf_assets(self, asset_type: Optional[str] = None) -> List[Asset]:
        """
        Return non-leaf-node assets (assets with at least one child).
        """
        Child = aliased(Asset)

        query = (
            self.session.query(Asset)
            .join(Child, Child.parent_uuid == Asset.asset_uuid)
            .distinct()
        )

        if asset_type:
            query = query.filter(Asset.asset_type == asset_type)

        return query.all()

    def list_non_leaf_assets_at_depth(
        self, depth: int, asset_type: Optional[str] = None
    ) -> List[Asset]:
        """
        Return non-leaf assets at a specific depth.
        """
        Child = aliased(Asset)

        query = (
            self.session.query(Asset)
            .join(Child, Child.parent_uuid == Asset.asset_uuid)
            .filter(Asset.depth == depth)
            .distinct()
            .order_by(Asset.asset_uuid.asc())
        )

        if asset_type:
            query = query.filter(Asset.asset_type == asset_type)

        return query.all()

    def get_max_depth_non_leaf(self, asset_type: Optional[str] = None) -> int:
        """
        Return the maximum depth among non-leaf assets.
        """
        Child = aliased(Asset)

        query = self.session.query(func.max(Asset.depth)).join(
            Child, Child.parent_uuid == Asset.asset_uuid
        )

        if asset_type:
            query = query.filter(Asset.asset_type == asset_type)

        return query.scalar() or 0

    def update_weather_site_id(self, asset_uuid: str, weather_site_id: str) -> None:
        """
        Update the weather_site_id for a given asset.

        Args:
            asset_uuid: The UUID of the asset to update
            weather_site_id: The weather site ID to assign
        """
        asset = self.get_asset(asset_uuid)
        if not asset:
            raise ValueError(f"Asset {asset_uuid} not found")

        asset.weather_site_id = weather_site_id
        self.session.commit()

    def list_assets_with_coords(
        self,
        asset_type: Optional[str] = None,
        missing_weather_site_only: bool = False,
    ) -> List[Asset]:
        query = (
            self.session.query(Asset)
            .filter(Asset.latitude.isnot(None))
            .filter(Asset.longitude.isnot(None))
        )

        if asset_type:
            query = query.filter(Asset.asset_type == asset_type)

        if missing_weather_site_only:
            query = query.filter(Asset.weather_site_id.is_(None))

        return query.all()
