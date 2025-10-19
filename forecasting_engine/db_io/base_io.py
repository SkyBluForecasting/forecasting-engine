from abc import ABC, abstractmethod
from sqlalchemy.orm import Session
import pandas as pd


class BaseIO(ABC):
    """Base class for DB I/O operations."""

    def __init__(self, session: Session):
        self.session = session

    @abstractmethod
    def to_df(self, *args, **kwargs) -> pd.DataFrame:
        """Read from DB → DataFrame"""
        pass  # pragma: no cover

    @abstractmethod
    def from_df(self, df: pd.DataFrame, *args, **kwargs):
        """Write DataFrame → DB"""
        pass  # pragma: no cover
