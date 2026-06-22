"""Strategy registry. Import lazily so a broken strategy doesn't sink the
whole harness; run.py looks strategies up by name."""

from importlib import import_module

_MODULES = {
    "s0": "seat_bench.strategies.s0_point",
    "s1": "seat_bench.strategies.s1_bbox_select",
    "s2": "seat_bench.strategies.s2_zoom",
    "s3": "seat_bench.strategies.s3_som",
}


def get_strategy(name: str):
    if name not in _MODULES:
        raise ValueError(f"unknown strategy {name!r}; choices={list(_MODULES)}")
    return import_module(_MODULES[name]).run


def all_strategy_names() -> list[str]:
    return list(_MODULES)
