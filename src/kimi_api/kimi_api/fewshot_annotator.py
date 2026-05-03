"""Web UI for annotating seat-pointing few-shot examples.

Run:

    ros2 run kimi_api seat_fewshot_annotator --image /abs/path/to/scene.jpg

The browser opens; the user enumerates `visible_seats`, picks a recommended
`label`, and clicks the recommended cushion to set `point` (in [y, x] over
0..1000, matching the Gemini structured-output schema in
``kimi_api/_seat_vlm.py``). On Save the JSON + a copy of the source image
are written into the package's source-tree `fewshot/<slug>/` directory.

Workflow under pure (non-symlink) colcon:

  1. Annotate → Save: writes go to the source tree (this file's enclosing
     package source dir, located via ``__file__`` since ament_python uses
     a develop/egg-link install).
  2. ``colcon build --packages-select kimi_api`` copies the new files into
     ``install/kimi_api/share/kimi_api/fewshot/`` via the ``data_files`` glob
     in ``setup.py``.
  3. Source ``install/setup.{bash,zsh}`` and the runtime loader picks them up.

Stdlib only — no Flask, no JS frameworks, no CDN.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import mimetypes
import os
import shutil
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

from ._seat_fewshot import _validate_answer  # reuse schema check


_PKG = 'kimi_api'
_FEWSHOT_DIRNAME = 'fewshot'


def _resolve_source_fewshot_dir() -> str:
    """Locate the source-tree fewshot directory.

    Under pure (non-symlink) ``colcon build`` of an ament_python package the
    layout is:

      - ``__file__`` for this module resolves to the **build** copy
        (``build/kimi_api/kimi_api/fewshot_annotator.py``) because Python
        sources are copied into ``build/`` and registered via an egg-link
        in ``install/.../site-packages/``.
      - Most files in ``build/<pkg>/`` are real copies, **but** ``package.xml``
        is symlinked back to source — a colcon convention that lets ament
        track package metadata changes without rebuild churn.

    So: from the build-side ``package.xml`` next to our build-side
    ``__file__``, follow its symlink via ``realpath``. The directory of the
    resolved file is the source package root, and ``fewshot/`` is a sibling.

    Running directly from source (``python <source>/.../fewshot_annotator.py``)
    also works — ``package.xml`` is then a regular file and ``realpath`` is
    a no-op. The sanity guard then refuses any path under ``install/`` or
    ``build/`` so a misconfigured workspace fails loud instead of silently
    wiping saves on the next rebuild.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_xml = os.path.join(here, 'package.xml')
    if not os.path.isfile(pkg_xml):
        raise SystemExit(
            f'Could not find package.xml at {pkg_xml} — package layout '
            'unexpected. Expected the build/source pkg root to contain '
            'package.xml + kimi_api/<this file>.'
        )
    src_pkg_root = os.path.dirname(os.path.realpath(pkg_xml))
    norm = os.path.normpath(src_pkg_root) + os.sep
    if any(seg in norm for seg in (os.sep + 'install' + os.sep,
                                   os.sep + 'build' + os.sep)):
        raise SystemExit(
            f'Resolved kimi_api source root ({src_pkg_root}) is inside '
            'install/ or build/. Saved examples would be wiped on the next '
            'clean rebuild. This means colcon failed to symlink package.xml '
            'back to source — re-run `colcon build --packages-select '
            f'{_PKG}` from {os.path.dirname(os.path.dirname(here))} '
            '(workspace root).'
        )
    fewshot_dir = os.path.join(src_pkg_root, _FEWSHOT_DIRNAME)
    os.makedirs(fewshot_dir, exist_ok=True)
    return fewshot_dir


def _slug_default(image_path: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    base = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base)
    ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{base}_{ts}'


def _is_valid_slug(slug: str) -> bool:
    if not slug or slug.startswith('.'):
        return False
    return all(c.isalnum() or c in ('-', '_') for c in slug)


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Seat-pointing few-shot annotator</title>
<style>
  body { margin: 0; font-family: system-ui, sans-serif; display: flex; height: 100vh; }
  #left { flex: 1; background: #111; display: flex; align-items: center; justify-content: center; overflow: auto; }
  #canvas { background: #222; cursor: crosshair; max-width: 100%; max-height: 100%; }
  #right { width: 420px; padding: 16px; overflow-y: auto; box-sizing: border-box; border-left: 1px solid #ddd; }
  h2 { margin: 4px 0 8px; }
  .row { display: flex; gap: 6px; margin-bottom: 6px; align-items: center; }
  .row input[type=text] { flex: 1; padding: 4px; }
  .row button { padding: 2px 8px; }
  .seat-row { padding: 6px; border: 1px solid #ccc; border-radius: 4px; margin-bottom: 6px; }
  .seat-row input[type=text] { width: 100%; padding: 4px; box-sizing: border-box; margin-bottom: 4px; }
  .seat-row label { font-size: 12px; }
  .seat-row .del { float: right; }
  .field { margin: 8px 0; }
  .field label { display: block; font-weight: 600; font-size: 12px; margin-bottom: 4px; }
  .field input, .field select { width: 100%; padding: 4px; box-sizing: border-box; }
  #status { white-space: pre-wrap; font-family: monospace; font-size: 12px; padding: 8px; background: #f4f4f4; border-radius: 4px; min-height: 20px; }
  .err { color: #b00; }
  .ok { color: #060; }
  button.primary { background: #06f; color: white; border: 0; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-size: 14px; }
  button.primary:disabled { background: #aaa; }
  .hint { font-size: 11px; color: #666; }
</style>
</head>
<body>
<div id="left">
  <canvas id="canvas"></canvas>
</div>
<div id="right">
  <h2>Visible seats</h2>
  <p class="hint">One row per cushion / single-person seat. Reasons should be short.</p>
  <div id="seats"></div>
  <button id="add-seat">+ add seat</button>

  <div class="field">
    <label for="label">Recommendation label</label>
    <select id="label"></select>
    <p class="hint">Pick from the seats above, or "none" if every seat is occupied.</p>
  </div>

  <div class="field">
    <label>Point</label>
    <span id="point-info" class="hint">click on the canvas</span>
  </div>

  <div class="field">
    <label for="slug">Slug (folder name)</label>
    <input id="slug" type="text">
  </div>

  <div class="field">
    <label for="notes">Notes (optional)</label>
    <input id="notes" type="text">
  </div>

  <button id="save" class="primary">Save</button>
  <pre id="status"></pre>
</div>

<script id="cfg-data" type="application/json">__CFG__</script>
<script>
(() => {
  const cfg = JSON.parse(document.getElementById('cfg-data').textContent);
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const seatsDiv = document.getElementById('seats');
  const labelSel = document.getElementById('label');
  const pointInfo = document.getElementById('point-info');
  const slugInput = document.getElementById('slug');
  const notesInput = document.getElementById('notes');
  const status = document.getElementById('status');
  slugInput.value = cfg.default_slug;

  let img = new Image();
  let pointPx = null;  // {x, y} in image pixel coords
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    redraw();
  };
  img.src = '/image';

  function redraw() {
    ctx.drawImage(img, 0, 0);
    if (pointPx) {
      ctx.strokeStyle = '#0f0';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(pointPx.x - 14, pointPx.y);
      ctx.lineTo(pointPx.x + 14, pointPx.y);
      ctx.moveTo(pointPx.x, pointPx.y - 14);
      ctx.lineTo(pointPx.x, pointPx.y + 14);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(pointPx.x, pointPx.y, 6, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  canvas.addEventListener('click', (ev) => {
    if (labelSel.value === 'none') return;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    pointPx = {
      x: Math.round((ev.clientX - rect.left) * sx),
      y: Math.round((ev.clientY - rect.top) * sy),
    };
    pointInfo.textContent = `pixel (${pointPx.x}, ${pointPx.y}) of ${canvas.width}x${canvas.height}`;
    redraw();
  });

  function refreshLabelDropdown() {
    const seats = collectSeats();
    const prev = labelSel.value;
    labelSel.innerHTML = '';
    for (const s of seats) {
      if (!s.label) continue;
      const opt = document.createElement('option');
      opt.value = s.label;
      opt.textContent = s.label;
      labelSel.appendChild(opt);
    }
    const noneOpt = document.createElement('option');
    noneOpt.value = 'none';
    noneOpt.textContent = 'none (every seat occupied)';
    labelSel.appendChild(noneOpt);
    if ([...labelSel.options].some(o => o.value === prev)) labelSel.value = prev;
  }

  function addSeatRow(label = '', occupied = false, reason = '') {
    const div = document.createElement('div');
    div.className = 'seat-row';
    div.innerHTML = `
      <button class="del" type="button">remove</button>
      <input type="text" placeholder="label (e.g. left cushion of gray sofa)" data-k="label">
      <input type="text" placeholder="reason (e.g. cushion clear)" data-k="reason">
      <label><input type="checkbox" data-k="occupied"> occupied</label>
    `;
    div.querySelector('[data-k=label]').value = label;
    div.querySelector('[data-k=reason]').value = reason;
    div.querySelector('[data-k=occupied]').checked = occupied;
    div.querySelector('.del').addEventListener('click', () => {
      div.remove();
      refreshLabelDropdown();
    });
    div.querySelector('[data-k=label]').addEventListener('input', refreshLabelDropdown);
    seatsDiv.appendChild(div);
    refreshLabelDropdown();
  }

  function collectSeats() {
    return [...seatsDiv.querySelectorAll('.seat-row')].map(div => ({
      label: div.querySelector('[data-k=label]').value.trim(),
      occupied: div.querySelector('[data-k=occupied]').checked,
      reason: div.querySelector('[data-k=reason]').value.trim(),
    }));
  }

  document.getElementById('add-seat').addEventListener('click', () => addSeatRow());
  labelSel.addEventListener('change', () => {
    if (labelSel.value === 'none') {
      pointPx = null;
      pointInfo.textContent = 'forced to (0, 0) for label="none"';
      redraw();
    } else if (!pointPx) {
      pointInfo.textContent = 'click on the canvas';
    }
  });

  document.getElementById('save').addEventListener('click', async () => {
    status.textContent = '';
    status.className = '';
    const seats = collectSeats();
    if (seats.length === 0) {
      status.textContent = 'Add at least one visible seat.';
      status.className = 'err';
      return;
    }
    for (const s of seats) {
      if (!s.label || !s.reason) {
        status.textContent = 'Every seat needs both a label and a reason.';
        status.className = 'err';
        return;
      }
    }
    const seatLabels = new Set(seats.map(s => s.label));
    const label = labelSel.value;
    let pointArr;
    if (label === 'none') {
      pointArr = [0, 0];
    } else {
      if (!seatLabels.has(label)) {
        status.textContent = 'Recommendation label must match one of the seats.';
        status.className = 'err';
        return;
      }
      if (!pointPx) {
        status.textContent = 'Click on the canvas to set the recommended point.';
        status.className = 'err';
        return;
      }
      const py = Math.round((pointPx.y / canvas.height) * 1000);
      const px = Math.round((pointPx.x / canvas.width) * 1000);
      pointArr = [Math.max(0, Math.min(1000, py)), Math.max(0, Math.min(1000, px))];
    }
    const slug = slugInput.value.trim();
    if (!slug) {
      status.textContent = 'Slug is required.';
      status.className = 'err';
      return;
    }
    const body = {
      slug,
      notes: notesInput.value,
      answer: { visible_seats: seats, label, point: pointArr },
    };
    status.textContent = 'Saving...';
    try {
      const r = await fetch('/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (r.ok && data.ok) {
        status.textContent =
          `Saved to ${data.path}\nNote: run \`${data.rebuild_hint}\` to register this example at runtime.`;
        status.className = 'ok';
      } else {
        status.textContent = `Error: ${data.error || r.status}`;
        status.className = 'err';
      }
    } catch (e) {
      status.textContent = `Network error: ${e}`;
      status.className = 'err';
    }
  });

  addSeatRow();
})();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):

    server_version = 'SeatFewshotAnnotator/0.1'

    # injected by main()
    image_path: str = ''
    fewshot_dir: str = ''
    default_slug: str = ''

    def log_message(self, fmt, *args):  # noqa: A003 — http.server convention
        sys.stderr.write('[annotator] ' + fmt % args + '\n')

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 — http.server convention
        if self.path in ('/', '/index.html'):
            cfg = {'default_slug': self.default_slug}
            html = _INDEX_HTML.replace('__CFG__', json.dumps(cfg))
            self._send_text(200, html.encode('utf-8'), 'text/html; charset=utf-8')
            return
        if self.path == '/image':
            ctype, _ = mimetypes.guess_type(self.image_path)
            if ctype is None:
                ctype = 'application/octet-stream'
            with open(self.image_path, 'rb') as f:
                data = f.read()
            self._send_text(200, data, ctype)
            return
        self._send_json(404, {'error': 'not found'})

    def do_POST(self):  # noqa: N802 — http.server convention
        if self.path != '/save':
            self._send_json(404, {'error': 'not found'})
            return
        length = int(self.headers.get('Content-Length', '0'))
        try:
            payload = json.loads(self.rfile.read(length).decode('utf-8'))
        except json.JSONDecodeError as exc:
            self._send_json(400, {'error': f'invalid JSON: {exc}'})
            return

        slug = payload.get('slug', '').strip()
        if not _is_valid_slug(slug):
            self._send_json(400, {'error': 'slug must be alnum/_/- and non-empty'})
            return

        answer = payload.get('answer')
        if not _validate_answer(answer):
            self._send_json(400, {'error': 'answer.json failed schema validation'})
            return

        slug_dir = os.path.join(self.fewshot_dir, slug)
        if os.path.exists(slug_dir):
            self._send_json(409, {'error': f'slug "{slug}" already exists'})
            return

        try:
            os.makedirs(slug_dir, exist_ok=False)
            ext = os.path.splitext(self.image_path)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png'):
                ext = '.jpg'
            image_dst = os.path.join(slug_dir, f'image{ext}')
            shutil.copyfile(self.image_path, image_dst)
            with open(os.path.join(slug_dir, 'answer.json'), 'w') as f:
                json.dump(answer, f, indent=2)
                f.write('\n')
            meta = {
                'created': _dt.datetime.utcnow().isoformat() + 'Z',
                'source_image': os.path.abspath(self.image_path),
                'notes': str(payload.get('notes', '')),
            }
            with open(os.path.join(slug_dir, 'meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
                f.write('\n')
        except OSError as exc:
            shutil.rmtree(slug_dir, ignore_errors=True)
            self._send_json(500, {'error': f'write failed: {exc}'})
            return

        self._send_json(200, {
            'ok': True,
            'slug': slug,
            'path': slug_dir,
            'rebuild_hint': (
                f'colcon build --packages-select {_PKG} '
                '&& source install/setup.zsh   # copies the new example into '
                'share/kimi_api/fewshot/ for the runtime loader'
            ),
        })


def main(argv=None):
    parser = argparse.ArgumentParser(prog='seat_fewshot_annotator')
    parser.add_argument('--image', required=True, help='Source image to annotate.')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--slug', default=None, help='Override default slug name.')
    args = parser.parse_args(argv)

    if not os.path.isfile(args.image):
        print(f'error: --image {args.image} does not exist', file=sys.stderr)
        return 2

    fewshot_dir = _resolve_source_fewshot_dir()
    default_slug = args.slug or _slug_default(args.image)

    _Handler.image_path = os.path.abspath(args.image)
    _Handler.fewshot_dir = fewshot_dir
    _Handler.default_slug = default_slug

    try:
        httpd = HTTPServer(('127.0.0.1', args.port), _Handler)
    except OSError as exc:
        print(f'error: could not bind 127.0.0.1:{args.port}: {exc}', file=sys.stderr)
        return 2

    url = f'http://127.0.0.1:{args.port}/'
    print(f'[annotator] serving {url}', file=sys.stderr)
    print(f'[annotator] image       = {_Handler.image_path}', file=sys.stderr)
    print(f'[annotator] fewshot_dir = {fewshot_dir}', file=sys.stderr)
    print(f'[annotator] default slug= {default_slug}', file=sys.stderr)
    print('[annotator] Ctrl-C to quit', file=sys.stderr)

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n[annotator] bye', file=sys.stderr)
    finally:
        httpd.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
