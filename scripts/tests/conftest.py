"""Pytest configuration for vision_util tests."""
import sys
from pathlib import Path

# Add src/vision_util to path so we pick up the local version, not the installed one
worktree_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(worktree_root / "src" / "vision_util"))
