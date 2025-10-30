import sys
from unittest.mock import MagicMock

# Mock DB imports globally for all tests
sys.modules["forecasting_engine.db_io.session"] = MagicMock()
