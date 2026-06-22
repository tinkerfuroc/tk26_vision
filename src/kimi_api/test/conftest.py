"""Pytest configuration: ensure the local source tree shadows the colcon-installed copy."""
import sys
from pathlib import Path

# Insert the package source dir so imports resolve against the src tree, not
# the stale colcon install under tk25_ws/install/ (which lags until rebuild).
_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
