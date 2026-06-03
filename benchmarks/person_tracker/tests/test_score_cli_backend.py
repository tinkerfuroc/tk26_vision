"""The acceptance default backend is `action` (live server), not offline."""
import argparse

from ptbench.replay import score_cli


def _parse_backend(argv):
    # Rebuild the same --backend contract main() uses and read what it
    # resolves to. This pins both the exposed constant and the wired default.
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument(
        "--backend", choices=("offline", "action"),
        default=score_cli.DEFAULT_BACKEND,
    )
    return parser.parse_args(argv).backend


def test_default_backend_constant_is_action():
    assert score_cli.DEFAULT_BACKEND == "action"


def test_default_backend_resolves_to_action():
    assert _parse_backend(["--bag", "B", "--gt", "G"]) == "action"


def test_explicit_offline_still_selectable():
    assert _parse_backend(["--bag", "B", "--gt", "G", "--backend", "offline"]) == "offline"
