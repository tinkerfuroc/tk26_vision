"""Generate a print-ready ChArUco board image + PDF.

Usage (standalone):
    # Named preset (recommended)
    python -m pan_tilt.calibration.charuco_generate --preset default --out /tmp/charuco
    python -m pan_tilt.calibration.charuco_generate --preset compact --out /tmp/charuco

    # Explicit overrides
    python -m pan_tilt.calibration.charuco_generate \\
        --squares-x 5 --squares-y 7 \\
        --square-len 0.040 --marker-len 0.030 \\
        --out /tmp/charuco

Presets:
    default  5x7 @ 40mm squares, 30mm markers -> 200x280 mm board
             Fills more of the camera frame; best accuracy at 0.5-1.5 m range.
             Tight A4 margins (~5mm / ~8mm).
    compact  5x5 @ 20mm squares, 15mm markers -> 100x100 mm board
             Fits a small EE-mount fixture (10x10 cm). Comfortable A4
             margins. Use when the physical EE real-estate is limited.

Outputs:
    <out>.png   - high-res board image (300 DPI at the target physical size)
    <out>.pdf   - A4 PDF with the board centered at exact scale
    <out>.json  - board spec, for programmatic re-instantiation at detection time

Print the PDF on **matte A4**, mount on 3 mm aluminium composite, and re-measure
the printed square size with calipers before trusting the extrinsic calibration
— even "100%" print settings warp slightly.

Whichever preset you use, update the `board:` section of calibration.yaml to
match so the collector's detector is instantiated with the correct spec.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .aruco_detect import BoardSpec, build_board


DPI = 300
INCH_PER_M = 39.3700787


def board_image(spec: BoardSpec, dpi: int = DPI) -> np.ndarray:
    """Render the board at the given DPI (pixels per inch)."""
    width_m = spec.squares_x * spec.square_len_m
    height_m = spec.squares_y * spec.square_len_m
    width_px = int(round(width_m * INCH_PER_M * dpi))
    height_px = int(round(height_m * INCH_PER_M * dpi))
    board = build_board(spec)
    return board.generateImage((width_px, height_px))


def write_pdf(png_path: Path, pdf_path: Path, spec: BoardSpec) -> None:
    """Embed the board image into an A4 PDF at exact physical scale.

    We write a minimal hand-crafted PDF (no external deps) — the image is
    embedded as a raw JPEG stream and positioned so that the printed square
    size equals `spec.square_len_m` to within the printer's accuracy.
    """
    import io

    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read {png_path}")

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    jpeg_bytes = buf.tobytes()

    # A4 portrait: 210mm x 297mm. PDF unit = 1/72 inch.
    PAGE_W_PT = 210.0 / 25.4 * 72.0
    PAGE_H_PT = 297.0 / 25.4 * 72.0

    board_w_pt = spec.squares_x * spec.square_len_m * INCH_PER_M * 72.0
    board_h_pt = spec.squares_y * spec.square_len_m * INCH_PER_M * 72.0

    if board_w_pt > PAGE_W_PT or board_h_pt > PAGE_H_PT:
        raise ValueError(
            f"Board {board_w_pt / 72 * 25.4:.1f}x{board_h_pt / 72 * 25.4:.1f}mm "
            f"exceeds A4 (210x297mm); reduce squares or square_len "
            f"(e.g. --square-len 0.035 --marker-len 0.026 gives 175x245mm)."
        )

    x_pt = (PAGE_W_PT - board_w_pt) / 2.0
    y_pt = (PAGE_H_PT - board_h_pt) / 2.0

    # Minimal PDF: 5 objects (Catalog, Pages, Page, XObject Image, Contents).
    objs = [None]  # 1-indexed

    def add(body: bytes) -> int:
        objs.append(body)
        return len(objs) - 1

    image_obj = (
        f"<< /Type /XObject /Subtype /Image /Width {img.shape[1]} /Height {img.shape[0]}"
        f" /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode"
        f" /Length {len(jpeg_bytes)} >>\nstream\n"
    ).encode("ascii") + jpeg_bytes + b"\nendstream"
    image_id = add(image_obj)

    content_stream = (
        f"q\n{board_w_pt} 0 0 {board_h_pt} {x_pt} {y_pt} cm\n/Im0 Do\nQ\n"
    ).encode("ascii")
    contents_obj = (
        f"<< /Length {len(content_stream)} >>\nstream\n".encode("ascii")
        + content_stream
        + b"\nendstream"
    )
    contents_id = add(contents_obj)

    page_obj = (
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W_PT} {PAGE_H_PT}]"
        f" /Resources << /XObject << /Im0 {image_id} 0 R >> >>"
        f" /Contents {contents_id} 0 R >>"
    ).encode("ascii")
    page_id = add(page_obj)

    pages_obj = f"<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>".encode("ascii")
    pages_id = add(pages_obj)
    # Pages always object 2; fix up via reordering.
    # We assembled: [1]=image, [2]=contents, [3]=page, [4]=pages. Rewrite page parent.
    objs[page_id] = (
        f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_W_PT} {PAGE_H_PT}]"
        f" /Resources << /XObject << /Im0 {image_id} 0 R >> >>"
        f" /Contents {contents_id} 0 R >>"
    ).encode("ascii")

    catalog_obj = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii")
    catalog_id = add(catalog_obj)

    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    xref_offsets = [0]
    for i in range(1, len(objs)):
        xref_offsets.append(buffer.tell())
        buffer.write(f"{i} 0 obj\n".encode("ascii"))
        buffer.write(objs[i])
        buffer.write(b"\nendobj\n")

    xref_pos = buffer.tell()
    buffer.write(f"xref\n0 {len(objs)}\n".encode("ascii"))
    buffer.write(b"0000000000 65535 f \n")
    for off in xref_offsets[1:]:
        buffer.write(f"{off:010d} 00000 n \n".encode("ascii"))
    buffer.write(
        f"trailer\n<< /Size {len(objs)} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )

    pdf_path.write_bytes(buffer.getvalue())


PRESETS = {
    "default": dict(squares_x=5, squares_y=7, square_len=0.040, marker_len=0.030),
    "compact": dict(squares_x=5, squares_y=5, square_len=0.020, marker_len=0.015),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a ChArUco board for calibration.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="named geometry preset; overridden by explicit --squares-* / --*-len",
    )
    parser.add_argument("--squares-x", type=int, default=None)
    parser.add_argument("--squares-y", type=int, default=None)
    parser.add_argument("--square-len", type=float, default=None, help="square side, meters")
    parser.add_argument("--marker-len", type=float, default=None, help="marker side, meters")
    parser.add_argument("--dict", type=str, default="DICT_5X5_100")
    parser.add_argument("--out", type=Path, required=True, help="output path stem (no extension)")
    args = parser.parse_args(argv)

    preset_vals = PRESETS[args.preset] if args.preset else PRESETS["default"]
    squares_x = args.squares_x or preset_vals["squares_x"]
    squares_y = args.squares_y or preset_vals["squares_y"]
    square_len = args.square_len or preset_vals["square_len"]
    marker_len = args.marker_len or preset_vals["marker_len"]

    dict_id = getattr(cv2.aruco, args.dict)
    spec = BoardSpec(
        squares_x=squares_x,
        squares_y=squares_y,
        square_len_m=square_len,
        marker_len_m=marker_len,
        dict_id=dict_id,
    )

    out_stem = args.out
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    png_path = out_stem.with_suffix(".png")
    pdf_path = out_stem.with_suffix(".pdf")
    json_path = out_stem.with_suffix(".json")

    img = board_image(spec)
    cv2.imwrite(str(png_path), img)
    write_pdf(png_path, pdf_path, spec)

    json_path.write_text(json.dumps({
        "squares_x": spec.squares_x,
        "squares_y": spec.squares_y,
        "square_len_m": spec.square_len_m,
        "marker_len_m": spec.marker_len_m,
        "dict": args.dict,
    }, indent=2))

    print(f"Board image: {png_path}  ({img.shape[1]}x{img.shape[0]} px)")
    print(f"PDF:         {pdf_path}")
    print(f"JSON spec:   {json_path}")
    print()
    print(f"Physical size when printed at 100% scale: "
          f"{spec.squares_x * spec.square_len_m * 1000:.0f}mm x "
          f"{spec.squares_y * spec.square_len_m * 1000:.0f}mm "
          f"({spec.squares_x * spec.squares_y} squares, "
          f"{spec.n_inner_corners} inner ChArUco corners)")


if __name__ == "__main__":
    main()
