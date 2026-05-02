#!/usr/bin/env python3
"""Vision Log Viewer — auto-discover the freshest vision_log session and
serve a browser view of recent service calls.

Renders three card layouts:
  * feature_matching   — scene overlay + ref0..refN crops with Ref->Cand legend
  * feature_extraction — scene overlay + chosen-person crop + feature text
  * everything else    — single overlay with tag/branch caption

The active folder is re-detected on every request, so a fresh session
created while the viewer is running is picked up on the next refresh
without operator action.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from collections import defaultdict
from urllib.parse import urlencode

from flask import Flask, Response, abort, request, send_file


SESSION_RE = re.compile(r'^\d{8}_\d{6}$')
KIND_RE = r'(?:yolo_detection_all|yolo_detection|ref\d+|orig|overlay|req|crop)'
TS_RE = r'\d{8}_\d{6}_\d{3}'
FILENAME_RE = re.compile(
    rf'^(?P<stem>.+)_(?P<kind>{KIND_RE})_(?P<ts>{TS_RE})\.(?P<ext>[A-Za-z0-9]+)$'
)
LEGACY_RE = re.compile(
    rf'^(?P<kind>orig|overlay|req|crop)_(?P<ts>{TS_RE})\.(?P<ext>[A-Za-z0-9]+)$'
)
KNOWN_BRANCHES = (
    'feature_matching', 'feature_extraction',
    'seat_recommend_bbox', 'seat_recommend',
    'placing', 'detect_waving', 'tracking',
    'follow_head', 'person_track',
    'yolo_world', 'vlm_sam', 'yolo', 'none',
)
REFRESH_CHOICES = (0, 1, 3, 10)


def find_active_folder(base: str) -> str | None:
    try:
        cands = [
            (e.stat().st_mtime, e.name)
            for e in os.scandir(base)
            if e.is_dir() and SESSION_RE.match(e.name)
        ]
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if not cands:
        return None
    return max(cands, key=lambda p: p[0])[1]


def parse_filename(name: str) -> dict | None:
    m = FILENAME_RE.match(name)
    if m:
        return {'stem': m.group('stem'), 'kind': m.group('kind'),
                'ts': m.group('ts'), 'ext': m.group('ext')}
    m = LEGACY_RE.match(name)
    if m:
        return {'stem': '', 'kind': m.group('kind'),
                'ts': m.group('ts'), 'ext': m.group('ext')}
    return None


def split_tag_branch(stem: str) -> tuple[str, str]:
    for b in KNOWN_BRANCHES:
        if stem.endswith('_' + b):
            return stem[: -(len(b) + 1)], b
    return stem, ''


def collect_calls(folder: str, max_entries: int) -> list[dict]:
    by_key: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    for name in names:
        parsed = parse_filename(name)
        if parsed is None:
            continue
        key = (parsed['stem'], parsed['ts'])
        by_key[key][parsed['kind']] = os.path.join(folder, name)

    calls: list[dict] = []
    for (stem, ts), kinds in by_key.items():
        meta: dict | None = None
        req_path = kinds.get('req')
        if req_path:
            try:
                with open(req_path) as fp:
                    meta = json.load(fp)
            except (OSError, json.JSONDecodeError):
                meta = None
        tag = (meta or {}).get('tag') or ''
        branch = (meta or {}).get('branch') or ''
        if not (tag and branch):
            t, b = split_tag_branch(stem)
            tag = tag or t
            branch = branch or b
        calls.append({
            'stem': stem, 'ts': ts, 'kinds': kinds,
            'meta': meta, 'tag': tag, 'branch': branch,
        })

    calls.sort(key=lambda c: c['ts'], reverse=True)
    return calls[:max_entries]


# ──────────────────────────── HTML rendering ────────────────────────────


CSS = """
:root { color-scheme: dark; }
body { font: 14px/1.4 -apple-system,Segoe UI,sans-serif; margin: 0;
       background:#1a1a1a; color:#ddd; }
header { padding: 10px 16px; background:#222; border-bottom:1px solid #333;
         display:flex; gap:18px; align-items:center; flex-wrap:wrap; }
header .folder { font-family: monospace; font-size:13px; color:#9cf; }
header .empty { color:#f88; }
header .meta { color:#888; font-size:12px; margin-left:auto; }
main { padding: 12px; display:grid; gap:12px;
       grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); }
.card { background:#252525; border:1px solid #333; border-radius:6px;
        padding: 8px 10px 10px; }
.card.fm { grid-column: span 2; }
.card.fe { grid-column: span 2; }
.card h2 { margin:0 0 6px; font-size:13px; font-weight:600;
           display:flex; gap:6px; align-items:baseline; }
.card .branch { color:#9cf; font-family:monospace; font-size:12px; }
.card .ts { color:#666; font-family:monospace; font-size:11px; margin-left:auto; }
.card img { display:block; max-width:100%; border-radius:4px;
            background:#111; }
.fm-grid { display:grid; grid-template-columns: 2fr 1fr; gap:8px; }
.fm-refs { display:grid; gap:6px; align-content:start; }
.fm-ref { background:#1d1d1d; border:1px solid #333; border-radius:4px;
          padding:4px; }
.fm-ref img { max-height:120px; margin:auto; }
.fm-ref .cap { font-size:11px; color:#9cf; font-family:monospace;
               margin-top:3px; }
.fm-ref .feat { font-size:11px; color:#aaa; margin-top:2px; }
.fe-grid { display:grid; grid-template-columns: 2fr 1fr; gap:8px;
           align-items:start; }
.fe-grid .crop img { max-height:240px; }
.fe-feat { font-size:12px; color:#bbf; margin-top:6px;
           padding:6px; background:#1d1d1d; border-radius:4px;
           font-family:monospace; white-space:pre-wrap; }
.foot { margin-top:6px; font-size:11px; color:#888; display:flex;
        gap:10px; flex-wrap:wrap; align-items:baseline; }
.foot a { color:#69c; text-decoration:none; }
.foot a:hover { text-decoration:underline; }
.foot .err { color:#f88; }
.placeholder { color:#666; font-style:italic; padding:18px; text-align:center; }
"""


def img_url(path: str, base_abs: str) -> str | None:
    if not path:
        return None
    rel = os.path.relpath(path, base_abs)
    return '/img?' + urlencode({'path': rel})


def req_url(path: str, base_abs: str) -> str:
    rel = os.path.relpath(path, base_abs)
    return '/req?' + urlencode({'path': rel})


def fmt_timings(timings: dict | None) -> str:
    if not timings:
        return ''
    bits = []
    for k, v in timings.items():
        try:
            ms = float(v) * 1000.0
            bits.append(f'{html.escape(str(k))}: {ms:.0f}ms' if ms < 1000
                        else f'{html.escape(str(k))}: {float(v):.2f}s')
        except (TypeError, ValueError):
            bits.append(f'{html.escape(str(k))}: ?')
    return ' · '.join(bits)


def fmt_ts(ts: str) -> str:
    if len(ts) >= 18 and ts[8] == '_' and ts[15] == '_':
        return f'{ts[9:11]}:{ts[11:13]}:{ts[13:15]}.{ts[16:19]}'
    return ts


def render_card_fm(c: dict, base_abs: str) -> str:
    """Feature matching card: scene overlay + ref grid + Ref->Cand legend."""
    overlay = c['kinds'].get('overlay') or c['kinds'].get('orig')
    overlay_url = img_url(overlay, base_abs) if overlay else None
    meta = c.get('meta') or {}
    matches = meta.get('matches') or []
    feats = meta.get('features_text') or []

    ref_paths: list[tuple[int, str]] = []
    for k, p in c['kinds'].items():
        m = re.match(r'^ref(\d+)$', k)
        if m:
            ref_paths.append((int(m.group(1)), p))
    ref_paths.sort()

    # ref_idx -> cand_idx
    ref_to_cand = {m['ref']: m['cand'] for m in matches if isinstance(m, dict)
                   and 'ref' in m and 'cand' in m}

    refs_html = []
    if ref_paths:
        for idx, path in ref_paths:
            cap = (f'Ref {idx} → Cand {ref_to_cand[idx]}'
                   if idx in ref_to_cand else f'Ref {idx}')
            url = img_url(path, base_abs)
            feat = (html.escape(feats[idx])
                    if idx < len(feats) and feats[idx] else '')
            refs_html.append(
                f'<div class="fm-ref">'
                f'<img src="{url}" alt="ref{idx}">'
                f'<div class="cap">{html.escape(cap)}</div>'
                + (f'<div class="feat">{feat}</div>' if feat else '')
                + '</div>'
            )
    else:
        refs_html.append(
            '<div class="placeholder">no reference crops written '
            '(text-only mode)</div>'
        )

    # Footer extras: vlm_status, n_detections, timings
    vstat = meta.get('vlm_status')
    verr = meta.get('vlm_error_msg') or ''
    foot_bits = []
    if c['kinds'].get('req'):
        foot_bits.append(f'<a href="{req_url(c["kinds"]["req"], base_abs)}" '
                         f'target="_blank">json</a>')
    if 'n_detections' in meta:
        foot_bits.append(f'{meta["n_detections"]} detections')
    if vstat is not None:
        cls = 'err' if vstat else ''
        msg = f'vlm_status={vstat}'
        if verr:
            msg += f' ({html.escape(str(verr))})'
        foot_bits.append(f'<span class="{cls}">{msg}</span>')
    t = fmt_timings(meta.get('timings'))
    if t:
        foot_bits.append(t)

    overlay_html = (f'<img src="{overlay_url}" alt="overlay">'
                    if overlay_url else
                    '<div class="placeholder">no overlay</div>')
    return (
        f'<div class="card fm">'
        f'<h2><span class="tag">{html.escape(c["tag"])}</span>'
        f'<span class="branch">· {html.escape(c["branch"])}</span>'
        f'<span class="ts">{fmt_ts(c["ts"])}</span></h2>'
        f'<div class="fm-grid">'
        f'<div class="fm-overlay">{overlay_html}</div>'
        f'<div class="fm-refs">{"".join(refs_html)}</div>'
        f'</div>'
        f'<div class="foot">{" · ".join(foot_bits)}</div>'
        f'</div>'
    )


def render_card_fe(c: dict, base_abs: str) -> str:
    """Feature extraction card: overlay + crop + feature text."""
    overlay = c['kinds'].get('overlay') or c['kinds'].get('orig')
    crop = c['kinds'].get('crop')
    meta = c.get('meta') or {}

    overlay_html = (f'<img src="{img_url(overlay, base_abs)}" alt="overlay">'
                    if overlay else
                    '<div class="placeholder">no overlay</div>')
    crop_html = (f'<img src="{img_url(crop, base_abs)}" alt="crop">'
                 if crop else
                 '<div class="placeholder">no crop</div>')

    feature = meta.get('feature') or ''
    feat_html = (f'<div class="fe-feat">{html.escape(feature)}</div>'
                 if feature else '')

    foot_bits = []
    if c['kinds'].get('req'):
        foot_bits.append(f'<a href="{req_url(c["kinds"]["req"], base_abs)}" '
                         f'target="_blank">json</a>')
    if 'n_persons_detected' in (meta.get('request') or {}):
        foot_bits.append(
            f'{meta["request"]["n_persons_detected"]} persons')
    if 'crop_size' in meta:
        cs = meta['crop_size']
        if isinstance(cs, list) and len(cs) == 2:
            foot_bits.append(f'crop {cs[0]}×{cs[1]}')
    vstat = meta.get('vlm_status')
    if vstat is not None and vstat != 0:
        verr = meta.get('vlm_error_msg') or ''
        foot_bits.append(
            f'<span class="err">vlm_status={vstat} '
            f'{html.escape(str(verr))}</span>')
    t = fmt_timings(meta.get('timings'))
    if t:
        foot_bits.append(t)

    return (
        f'<div class="card fe">'
        f'<h2><span class="tag">{html.escape(c["tag"])}</span>'
        f'<span class="branch">· {html.escape(c["branch"])}</span>'
        f'<span class="ts">{fmt_ts(c["ts"])}</span></h2>'
        f'<div class="fe-grid">'
        f'<div class="overlay">{overlay_html}</div>'
        f'<div class="crop">{crop_html}</div>'
        f'</div>'
        f'{feat_html}'
        f'<div class="foot">{" · ".join(foot_bits)}</div>'
        f'</div>'
    )


def render_card_default(c: dict, base_abs: str) -> str:
    """Overlay-only card."""
    overlay = (c['kinds'].get('overlay') or c['kinds'].get('orig')
               or next(iter(c['kinds'].values()), None))
    meta = c.get('meta') or {}

    img_html = (f'<img src="{img_url(overlay, base_abs)}" alt="overlay">'
                if overlay else
                '<div class="placeholder">no image</div>')

    foot_bits = []
    if c['kinds'].get('req'):
        foot_bits.append(f'<a href="{req_url(c["kinds"]["req"], base_abs)}" '
                         f'target="_blank">json</a>')
    if 'n_detections' in meta:
        foot_bits.append(f'{meta["n_detections"]} det')
    t = fmt_timings(meta.get('timings'))
    if t:
        foot_bits.append(t)

    branch_html = (f'<span class="branch">· {html.escape(c["branch"])}</span>'
                   if c['branch'] else '')
    return (
        f'<div class="card">'
        f'<h2><span class="tag">{html.escape(c["tag"] or "(legacy)")}</span>'
        f'{branch_html}'
        f'<span class="ts">{fmt_ts(c["ts"])}</span></h2>'
        f'{img_html}'
        f'<div class="foot">{" · ".join(foot_bits)}</div>'
        f'</div>'
    )


def render_card(c: dict, base_abs: str) -> str:
    branch = c['branch']
    if branch == 'feature_matching':
        return render_card_fm(c, base_abs)
    if branch == 'feature_extraction':
        return render_card_fe(c, base_abs)
    return render_card_default(c, base_abs)


def render_page(folder_name: str | None, calls: list[dict],
                refresh_sec: int, base_abs: str, base_arg: str,
                max_entries: int) -> str:
    refresh_meta = (f'<meta http-equiv="refresh" content="{refresh_sec}">'
                    if refresh_sec > 0 else '')
    options = ''.join(
        f'<option value="{n}"{" selected" if n == refresh_sec else ""}>'
        f'{"off" if n == 0 else f"{n}s"}</option>'
        for n in REFRESH_CHOICES
    )
    selector = (
        f'<form method="get" style="margin:0">'
        f'<label>refresh: <select name="refresh" onchange="this.form.submit()">'
        f'{options}</select></label></form>'
    )

    if folder_name:
        head = (f'<span class="folder">{html.escape(folder_name)}</span>'
                f' <span style="color:#888">'
                f'({len(calls)} calls, max {max_entries})</span>')
    else:
        head = (f'<span class="empty">no sessions found in '
                f'{html.escape(base_arg)}</span>')

    if calls:
        body = ''.join(render_card(c, base_abs) for c in calls)
    elif folder_name:
        body = ('<div class="placeholder">folder is empty — waiting for '
                'first vision call…</div>')
    else:
        body = ('<div class="placeholder">create a '
                '<code>YYYYmmdd_HHMMSS/</code> subdir under '
                f'<code>{html.escape(base_arg)}</code> '
                'and the viewer will pick it up</div>')

    now_str = time.strftime('%H:%M:%S')

    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<title>vision_log viewer</title>'
        f'{refresh_meta}'
        f'<style>{CSS}</style></head><body>'
        f'<header>{head}{selector}'
        f'<span class="meta">refreshed {now_str}</span></header>'
        f'<main>{body}</main>'
        '</body></html>'
    )


# ──────────────────────────── Flask app ────────────────────────────


def create_app(base: str, max_entries: int) -> Flask:
    app = Flask(__name__)
    base_abs = os.path.realpath(base)

    def safe_resolve(rel: str) -> str:
        if not rel or rel.startswith('/') or '..' in rel.split('/'):
            abort(403)
        full = os.path.realpath(os.path.join(base_abs, rel))
        if not (full == base_abs or full.startswith(base_abs + os.sep)):
            abort(403)
        if not os.path.isfile(full):
            abort(404)
        return full

    @app.route('/')
    def index() -> Response:
        try:
            refresh_sec = int(request.args.get('refresh', '3'))
        except ValueError:
            refresh_sec = 3
        if refresh_sec not in REFRESH_CHOICES:
            refresh_sec = 3
        folder_name = find_active_folder(base_abs)
        if folder_name:
            calls = collect_calls(
                os.path.join(base_abs, folder_name), max_entries)
        else:
            calls = []
        html_doc = render_page(folder_name, calls, refresh_sec,
                               base_abs, base, max_entries)
        return Response(html_doc, mimetype='text/html',
                        headers={'Cache-Control': 'no-store'})

    @app.route('/img')
    def img() -> Response:
        full = safe_resolve(request.args.get('path', ''))
        resp = send_file(full)
        resp.headers['Cache-Control'] = 'no-store'
        return resp

    @app.route('/req')
    def req() -> Response:
        full = safe_resolve(request.args.get('path', ''))
        try:
            with open(full) as fp:
                payload = json.load(fp)
            body = json.dumps(payload, indent=2, sort_keys=False, default=str)
        except (OSError, json.JSONDecodeError) as e:
            body = f'(could not parse: {e})'
        page = (
            '<!doctype html><html><head><meta charset="utf-8">'
            f'<title>{html.escape(os.path.basename(full))}</title>'
            f'<style>{CSS} pre{{padding:14px;background:#111;'
            'border-radius:4px;overflow:auto}}</style></head>'
            f'<body><header><span class="folder">'
            f'{html.escape(os.path.basename(full))}</span></header>'
            f'<main><pre>{html.escape(body)}</pre></main></body></html>'
        )
        return Response(page, mimetype='text/html',
                        headers={'Cache-Control': 'no-store'})

    return app


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='vision_log',
                   help='vision_log root (default: vision_log)')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--max-entries', type=int, default=30,
                   help='max number of calls to render per page')
    args = p.parse_args()

    if not os.path.isdir(args.base):
        print(f'note: --base {args.base!r} does not exist yet; the viewer '
              f'will keep checking and pick it up when it appears.',
              file=sys.stderr)

    app = create_app(args.base, args.max_entries)
    print(f'vision_log_viewer: http://{args.host}:{args.port}/  '
          f'(base={args.base})', file=sys.stderr)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
