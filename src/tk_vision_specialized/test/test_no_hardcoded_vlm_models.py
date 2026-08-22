"""Guard: no production code may hardcode a VLM model id — defaults come from
vision_util.vlm_models so .env controls them. Scans the vision source tree."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCAN_DIRS = [REPO / 'src', REPO / 'scripts']
ALLOWED = {
    REPO / 'src' / 'vision_util' / 'vision_util' / 'vlm_models.py',
}
SKIP_PARTS = {'test', 'tests', 'fixtures', 'thirdparty', 'seat_bench/report.md'}
LITERALS = re.compile(
    r"['\"](dashscope/)?(google/gemini-2\.5-(pro|flash)|qwen3-vl-plus|gemini-2\.5-flash)['\"]"
)


def _candidates():
    for root in SCAN_DIRS:
        for path in root.rglob('*.py'):
            if path in ALLOWED:
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            yield path


def test_no_literal_model_ids_outside_resolver():
    offenders = []
    for path in _candidates():
        for lineno, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            if LITERALS.search(line) and not line.lstrip().startswith('#'):
                offenders.append(f'{path.relative_to(REPO)}:{lineno}: {line.strip()}')
    assert not offenders, 'hardcoded VLM model ids (use vision_util.vlm_models):\n' + '\n'.join(offenders)
