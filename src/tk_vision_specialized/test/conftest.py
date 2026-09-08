"""Pytest configuration: ensure the local source tree shadows the colcon-installed copy."""
import sys
from pathlib import Path

# Insert the package source dir so imports resolve against the src tree, not
# the stale colcon install under tk25_ws/install/ (which lags until rebuild).
_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

# kimi_api is a sibling package this one imports from (e.g. resolve_qwen_target);
# insert its source dir too, for the same reason as _PKG_DIR above.
_KIMI_API_PKG_ROOT = _PKG_DIR.parent / 'kimi_api'
if str(_KIMI_API_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_KIMI_API_PKG_ROOT))
