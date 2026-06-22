"""Shared types + helpers for strategy runners.

Every strategy exposes:  run(img_bgr, req, provider, logger=None) -> Result
where `req` is the dict {names, features, known_seats} from the scene's
.req.json. Result is JSON-serializable via to_dict().
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Result:
    strategy: str
    provider: str
    chosen_label: str = "none"
    point_xy: Optional[list] = None          # [x, y] pixels or None
    box_xyxy: Optional[list] = None          # [x1,y1,x2,y2] pixels or None
    visible_seats: list = field(default_factory=list)
    n_calls: int = 0
    elapsed_s: float = 0.0
    error: Optional[str] = None
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_request_text(req: dict) -> str:
    """Replay the production request phrasing (mirrors _seat_vlm._build_text_prompt)."""
    names = req.get("names", []) or []
    features = req.get("features", []) or []
    known = req.get("known_seats", []) or []
    text = "Recommend a seat for a new guest."
    for name, feature in zip(names, features):
        text += f" The person matching description: {feature} is called {name}."
    if known:
        lines = "\n".join(f'  - "{s}"' for s in known)
        text += (
            "\n\nThe seats in this room are pre-catalogued. The recommendation "
            "label MUST be exactly one of these strings, or \"none\" if every "
            "catalogued seat is occupied or not visible:\n" + lines
        )
    return text
