#!/usr/bin/env python3
"""
colony_picker.py — Headless PIXL Colony Picker
================================================
Replicates the colony_picker_v4 HTML tool with default settings.
Designed for automated execution by Thermo Fisher Momentum.

Usage
-----
  # Single image, output to same folder as image:
  python colony_picker.py /path/to/Original.png

  # Specify output directory:
  python colony_picker.py /path/to/Original.png --output /path/to/results/

  # Find and process the most-recently-modified image in a PIXL run folder:
  python colony_picker.py --run-folder /path/to/PIXL/runs/ --output /path/to/results/

  # Override defaults:
  python colony_picker.py /path/to/Original.png --target 96 --sensitivity 7

Options
-------
  --output DIR          Output directory (default: same folder as image)
  --run-folder DIR      Auto-find newest image in this PIXL run folder tree
  --source-name NAME    Source plate name in CSV  (default: SourcePlate1)
  --target-name NAME    Target plate name in CSV  (default: TargetPlate1)
  --target-format FMT   DWP_96 | SBS_96 | DWP_384  (default: DWP_96)
  --target INT          Number of colonies to pick  (default: 88)
  --sensitivity INT     1–10, higher = more permissive  (default: 5)
  --min-diameter INT    Minimum colony diameter in px   (default: 8)
  --max-diameter INT    Maximum colony diameter in px   (default: 50)
  --min-isolation FLOAT Minimum neighbour distance mm   (default: 0.0)
  --min-circularity F   0–1, minimum blob circularity   (default: 0.0)
  --exclude WELLS       Comma-separated wells to skip   (default: none)
  --annotated-image     Also save an annotated PNG
  --verbose             Print progress to stdout
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Required packages missing. Run:  pip install Pillow numpy",
          file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must match HTML tool exactly)
# ─────────────────────────────────────────────────────────────────────────────

# Singer PIXL instrument geometry constant.
# LED outer-to-outer physical span (mm). Validated against image segmentation
# centroids of 8 matched colonies across two plates: mean error < 0.07 mm.
PIXL_LED_SPAN_MM = 137.614

# SBS plate maximum coordinate extents (mm from centre)
SBS_X_MAX = 63.88
SBS_Y_MAX = 42.74

# Quality scoring weights (circularity², size, isolation)
W_CIRC = 0.35
W_SIZE = 0.15
W_ISOL = 0.50

IDEAL_R_MM   = 0.8   # peak quality radius (mm)
HARD_CIRC    = 0.65  # blobs below this are discarded
MERGE_R_MM   = 1.5   # radius above which merged-blob penalty kicks in


# ─────────────────────────────────────────────────────────────────────────────
# PLATE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def detect_plate_calibration(gray: np.ndarray) -> Tuple[float, float, float, dict]:
    """
    Detect LED bar positions from the grayscale image and return:
        plate_cx, plate_cy  — image-pixel coordinates of plate centre
        px_per_mm           — pixels per millimetre (PIXL-calibrated)
        bounds              — dict with r1,r2,c1,c2 (inner crop used for detection)
    Uses the LED bar OUTER edges + PIXL_LED_SPAN_MM constant, matching the
    physical robot coordinate system.
    """
    H, W = gray.shape
    LED_THRESH = 120
    GAP = 5
    MARGIN_ROW = 80
    MARGIN_COL = 110

    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)

    def find_bands(means, length):
        bright = [i for i, m in enumerate(means) if m > LED_THRESH]
        if not bright:
            return []
        bands, s, p = [], bright[0], bright[0]
        for k in range(1, len(bright)):
            if bright[k] > p + GAP:
                bands.append((s, p))
                s = bright[k]
            p = bright[k]
        bands.append((s, p))
        return bands

    row_bands = find_bands(row_means, H)
    col_bands = find_bands(col_means, W)

    # Inner-edge crop (used as plate bounds for detection filtering)
    if len(row_bands) >= 2:
        r1 = row_bands[0][1]  + MARGIN_ROW
        r2 = row_bands[-1][0] - MARGIN_ROW
    elif len(row_bands) == 1:
        mid = (row_bands[0][0] + row_bands[0][1]) // 2
        r1, r2 = (row_bands[0][1]+MARGIN_ROW, H-MARGIN_ROW) if mid < H//2 \
                 else (MARGIN_ROW, row_bands[0][0]-MARGIN_ROW)
    else:
        r1, r2 = int(H*0.1), int(H*0.9)

    if len(col_bands) >= 2:
        c1 = col_bands[0][1]  + MARGIN_COL
        c2 = col_bands[-1][0] - MARGIN_COL
    elif len(col_bands) == 1:
        mid = (col_bands[0][0] + col_bands[0][1]) // 2
        c1, c2 = (col_bands[0][1]+MARGIN_COL, W-MARGIN_COL) if mid < W//2 \
                 else (MARGIN_COL, col_bands[0][0]-MARGIN_COL)
    else:
        c1, c2 = int(W*0.1), int(W*0.9)

    bounds = dict(r1=r1, r2=r2, c1=c1, c2=c2)

    # PIXL-calibrated centre and scale — use LED OUTER edges
    if len(col_bands) >= 2:
        outer_l = col_bands[0][0]
        outer_r = col_bands[-1][1]
        plate_cx  = (outer_l + outer_r) / 2.0
        px_per_mm = (outer_r - outer_l) / PIXL_LED_SPAN_MM
    else:
        plate_cx  = (c1 + c2) / 2.0
        px_per_mm = (c2 - c1) / 127.76

    if len(row_bands) >= 2:
        outer_t  = row_bands[0][0]
        outer_b  = row_bands[-1][1]
        plate_cy = (outer_t + outer_b) / 2.0
        # px_per_mm is isotropic — reuse X value
    else:
        plate_cy = (r1 + r2) / 2.0

    return plate_cx, plate_cy, px_per_mm, bounds


# ─────────────────────────────────────────────────────────────────────────────
# COLONY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def otsu_threshold(values: np.ndarray) -> int:
    hist = np.bincount(np.clip(values.astype(np.uint8).ravel(), 0, 255), minlength=256)
    total = values.size
    sum_t = np.dot(np.arange(256), hist)
    w_b = sum_b = best_var = 0
    otsu_t = 100
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_t - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, otsu_t = var, t
    return otsu_t


def connected_components(mask: np.ndarray) -> np.ndarray:
    """Union-Find connected components on a boolean mask. Returns label array."""
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    parent = list(range(H * W + 1))
    rank   = [0] * (H * W + 1)
    next_label = [1]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa == pb:
            return
        if rank[pa] < rank[pb]:
            parent[pa] = pb
        elif rank[pa] > rank[pb]:
            parent[pb] = pa
        else:
            parent[pb] = pa
            rank[pa] += 1

    for y in range(H):
        for x in range(W):
            if not mask[y, x]:
                continue
            above = labels[y-1, x] if y > 0 else 0
            left  = labels[y, x-1] if x > 0 else 0
            if not above and not left:
                lbl = next_label[0]
                next_label[0] += 1
                parent[lbl] = lbl
                labels[y, x] = lbl
            elif above and not left:
                labels[y, x] = find(above)
            elif not above and left:
                labels[y, x] = find(left)
            else:
                labels[y, x] = find(above)
                union(above, left)

    # Resolve labels
    for y in range(H):
        for x in range(W):
            if labels[y, x]:
                labels[y, x] = find(labels[y, x])
    return labels


def detect_colonies(
    img_array: np.ndarray,
    plate_cx: float,
    plate_cy: float,
    px_per_mm: float,
    bounds: dict,
    sensitivity: int = 5,
    min_diam_px: int = 8,
    max_diam_px: int = 50,
    verbose: bool = False,
) -> List[dict]:
    """
    Detect colony blobs and score them. Returns list of dicts sorted by quality.
    All coords are in full-image pixel space.
    """
    r1, r2, c1, c2 = bounds['r1'], bounds['r2'], bounds['c1'], bounds['c2']
    PH, PW = r2 - r1, c2 - c1

    # Build plate-only grayscale and red channel
    crop_rgb = img_array[r1:r2, c1:c2]
    gray = (crop_rgb[:,:,0] * 0.299 +
            crop_rgb[:,:,1] * 0.587 +
            crop_rgb[:,:,2] * 0.114).astype(np.float32)
    red_ch = crop_rgb[:,:,0].astype(np.float32)

    # Otsu threshold with sensitivity adjustment
    otsu_t = otsu_threshold(gray)
    adj_t  = otsu_t - (sensitivity - 5) * 4   # sens=5 → no change
    threshold = max(20, min(200, adj_t))
    if verbose:
        print(f"  Otsu={otsu_t}  adjusted={threshold}  (sensitivity={sensitivity})")

    # Foreground mask
    bg_brightness = float(gray.mean())
    if bg_brightness < 100:
        # Dark agar background — colonies are bright blobs
        fg = gray > threshold
    else:
        # Light background — use redness heuristic
        redness = red_ch / (gray * (0.587 / 0.299) + 1)
        fg = (redness > 1.05) & (gray > 80) & (gray < 220)

    if verbose:
        print(f"  Background brightness={bg_brightness:.1f}  "
              f"{'dark-agar' if bg_brightness < 100 else 'light-bg'} mode  "
              f"fg_pixels={fg.sum()}")

    # Connected components
    labels = connected_components(fg)

    # Collect blob statistics
    blob_map: dict = {}
    ys_all, xs_all = np.where(labels > 0)
    for y, x in zip(ys_all, xs_all):
        lbl = int(labels[y, x])
        if lbl not in blob_map:
            blob_map[lbl] = dict(n=0, sx=0, sy=0, x1=x, x2=x, y1=y, y2=y)
        b = blob_map[lbl]
        b['n'] += 1
        b['sx'] += x
        b['sy'] += y
        if x < b['x1']: b['x1'] = x
        if x > b['x2']: b['x2'] = x
        if y < b['y1']: b['y1'] = y
        if y > b['y2']: b['y2'] = y

    min_r = min_diam_px / 2.0
    max_r = max_diam_px / 2.0

    raw = []
    for b in blob_map.values():
        r_x = (b['x2'] - b['x1']) / 2.0
        r_y = (b['y2'] - b['y1']) / 2.0
        r   = (r_x + r_y) / 2.0
        if r < min_r or r > max_r:
            continue
        area  = b['n']
        perim = 2 * math.pi * r
        circ  = min((4 * math.pi * area) / (perim ** 2 + 1e-9), 1.0)
        cx_px = b['sx'] / b['n'] + c1   # back to full-image coordinates
        cy_px = b['sy'] / b['n'] + r1
        raw.append(dict(
            x=cx_px, y=cy_px, r=r,
            x_mm=(cx_px - plate_cx) / px_per_mm,
            y_mm=(cy_px - plate_cy) / px_per_mm,
            circularity=circ,
            area=area,
        ))

    # Filter to SBS plate working area
    raw = [b for b in raw
           if abs(b['x_mm']) <= SBS_X_MAX and abs(b['y_mm']) <= SBS_Y_MAX]

    if verbose:
        print(f"  Raw blobs inside plate bounds: {len(raw)}")

    # Nearest-neighbour edge distance for each blob
    pos  = [(b['x'], b['y']) for b in raw]
    rads = [b['r'] for b in raw]
    for i, b in enumerate(raw):
        min_edge = math.inf
        for j in range(len(raw)):
            if i == j:
                continue
            d = math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
            edge = d - rads[i] - rads[j]
            if edge < min_edge:
                min_edge = edge
        b['nn_mm'] = max(0.0, min_edge) / px_per_mm

    # Quality score (identical to HTML tool)
    for b in raw:
        if b['circularity'] < HARD_CIRC:
            b['quality'] = 0.0
            continue
        r_mm = b['r'] / px_per_mm
        if r_mm <= MERGE_R_MM:
            size_score = max(0.0, 1.0 - abs(r_mm - IDEAL_R_MM) / 1.5)
        else:
            size_score = max(0.0, 1.0 - (r_mm - MERGE_R_MM) * 1.5)
        circ_score = b['circularity'] ** 2
        b['quality'] = (circ_score * W_CIRC
                        + size_score * W_SIZE
                        + min(b['nn_mm'] / 2.0, 1.0) * W_ISOL)

    raw.sort(key=lambda b: b['quality'], reverse=True)
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# WELL ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def parse_exclusions(excl_str: str) -> set:
    out = set()
    if not excl_str:
        return out
    for part in excl_str.split(','):
        part = part.strip()
        rng = __import__('re').match(r'^([A-Za-z])(\d+)-\1(\d+)$', part)
        if rng:
            row = rng.group(1).upper()
            for c in range(int(rng.group(2)), int(rng.group(3)) + 1):
                out.add(row + str(c))
        else:
            m = __import__('re').match(r'^([A-Za-z])(\d+)$', part)
            if m:
                out.add(m.group(1).upper() + m.group(2))
    return out


def available_wells(target_format: str, exclude: set) -> List[str]:
    if '384' in target_format:
        rows, cols = 16, 24
    else:
        rows, cols = 8, 12
    wells = []
    for r in range(rows):
        for c in range(1, cols + 1):
            w = chr(65 + r) + str(c)
            if w not in exclude:
                wells.append(w)
    return wells


def select_colonies(
    all_blobs: List[dict],
    target: int,
    target_format: str,
    min_isolation: float = 0.0,
    min_circularity: float = 0.0,
    min_radius_px: float = 4.0,
    max_radius_px: float = 50.0,
    exclude_wells: set = None,
) -> List[dict]:
    """Apply quality filters and assign wells, exactly as the HTML tool does."""
    if exclude_wells is None:
        exclude_wells = set()

    passing = [
        b for b in all_blobs
        if (b['nn_mm']       >= min_isolation
            and b['circularity'] >= min_circularity
            and b['r']           >= min_radius_px
            and b['r']           <= max_radius_px)
    ]

    wells   = available_wells(target_format, exclude_wells)
    n_take  = min(len(passing), len(wells), target)
    selected = []
    for i in range(n_take):
        entry = dict(passing[i])
        entry['well'] = wells[i]
        entry['type'] = 'auto'
        selected.append(entry)
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(
    selected: List[dict],
    plate_cx: float,
    plate_cy: float,
    px_per_mm: float,
    source_name: str,
    target_name: str,
    target_format: str,
    output_path: Path,
):
    """Write rearray CSV in PIXL format (matches HTML exportCSV exactly)."""
    tt, tw = target_format.split('_')
    lines = [
        f"{source_name},SBS,None,Source,,",
        f"{target_name},{tt},{tw},Target,,",
    ]
    for c in selected:
        if not c.get('well'):
            continue
        m = __import__('re').match(r'^([A-Z])(\d+)$', c['well'])
        if not m:
            continue
        x_mm = (c['x'] - plate_cx) / px_per_mm
        y_mm = (c['y'] - plate_cy) / px_per_mm
        lines.append(
            f"{source_name},{y_mm:.8f},{x_mm:.8f},{target_name},{m.group(1)},{m.group(2)}"
        )
    output_path.write_text('\n'.join(lines) + '\n')


def export_annotated_image(
    img_array: np.ndarray,
    all_blobs: List[dict],
    selected: List[dict],
    bounds: dict,
    output_path: Path,
):
    """Save annotated PNG with detected and selected colonies marked."""
    img = Image.fromarray(img_array).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Plate boundary
    b = bounds
    draw.rectangle([b['c1'], b['r1'], b['c2'], b['r2']],
                   outline=(0, 229, 192), width=3)

    # All detected (faint teal)
    sel_keys = {(round(s['x']), round(s['y'])) for s in selected}
    for blob in all_blobs:
        if (round(blob['x']), round(blob['y'])) in sel_keys:
            continue
        q = min(blob['quality'] / 0.8, 1.0)
        if q < 0.05:
            continue
        r = int(blob['r'])
        alpha = int(q * 100)
        draw.ellipse(
            [blob['x']-r, blob['y']-r, blob['x']+r, blob['y']+r],
            outline=(0, 150, 130), width=1,
        )

    # Selected (orange with well label)
    for col in selected:
        r = max(int(col['r']), 8)
        draw.ellipse(
            [col['x']-r, col['y']-r, col['x']+r, col['y']+r],
            outline=(255, 112, 67), width=3,
        )
        draw.text(
            (col['x'] - r - 2, col['y'] - r - 14),
            col.get('well', '?'),
            fill=(255, 255, 255),
        )

    img.save(output_path, 'PNG')


def export_summary_json(
    all_blobs: List[dict],
    selected: List[dict],
    plate_cx: float,
    plate_cy: float,
    px_per_mm: float,
    bounds: dict,
    args,
    output_path: Path,
):
    summary = dict(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
        source_image=str(args.image) if args.image else None,
        source_name=args.source_name,
        target_name=args.target_name,
        target_format=args.target_format,
        target_count=args.target,
        settings=dict(
            sensitivity=args.sensitivity,
            min_diameter_px=args.min_diameter,
            max_diameter_px=args.max_diameter,
            min_isolation_mm=args.min_isolation,
            min_circularity=args.min_circularity,
            exclude_wells=args.exclude or '',
        ),
        calibration=dict(
            plate_cx=plate_cx,
            plate_cy=plate_cy,
            px_per_mm=px_per_mm,
            bounds=bounds,
            led_span_mm=PIXL_LED_SPAN_MM,
        ),
        results=dict(
            total_detected=len(all_blobs),
            total_selected=len(selected),
        ),
        colonies=[dict(
            well=c['well'],
            x_mm=round((c['x'] - plate_cx) / px_per_mm, 4),
            y_mm=round((c['y'] - plate_cy) / px_per_mm, 4),
            x_px=round(c['x']),
            y_px=round(c['y']),
            radius_px=round(c['r'], 1),
            circularity=round(c['circularity'], 3),
            isolation_mm=round(c['nn_mm'], 3),
            quality=round(c['quality'], 4),
        ) for c in selected],
    )
    output_path.write_text(json.dumps(summary, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# RUN-FOLDER AUTO-DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def find_newest_image(run_folder: Path) -> Optional[Path]:
    """
    Walk the run folder tree and return the most recently modified image file
    named 'Original*.png' (case-insensitive), matching PIXL naming convention.
    Falls back to any .png / .jpg if no Original* file is found.
    """
    candidates = []
    for ext in ('*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG'):
        candidates.extend(run_folder.rglob(ext))

    # Prefer files named Original* first
    originals = [p for p in candidates if p.stem.lower().startswith('original')]
    pool = originals if originals else candidates
    if not pool:
        return None
    return max(pool, key=lambda p: p.stat().st_mtime)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description='Headless PIXL colony picker — outputs rearray CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('image', nargs='?', type=Path,
                   help='Path to plate image (PNG/JPG). '
                        'Omit if using --run-folder.')
    p.add_argument('--run-folder', type=Path, metavar='DIR',
                   help='Auto-find the newest Original*.png in this PIXL run folder tree.')
    p.add_argument('--output', '-o', type=Path, metavar='DIR',
                   help='Output directory (default: same folder as image).')
    p.add_argument('--source-name', default='SourcePlate1',
                   help='Source plate name in CSV (default: SourcePlate1)')
    p.add_argument('--target-name', default='TargetPlate1',
                   help='Target plate name in CSV (default: TargetPlate1)')
    p.add_argument('--target-format', default='DWP_96',
                   choices=['DWP_96', 'SBS_96', 'DWP_384'],
                   help='Target plate format (default: DWP_96)')
    p.add_argument('--target', type=int, default=88, metavar='N',
                   help='Number of colonies to pick (default: 88)')
    p.add_argument('--sensitivity', type=int, default=5, choices=range(1, 11),
                   metavar='1-10',
                   help='Detection sensitivity 1–10 (default: 5)')
    p.add_argument('--min-diameter', type=int, default=8, metavar='PX',
                   help='Min colony diameter in px (default: 8)')
    p.add_argument('--max-diameter', type=int, default=50, metavar='PX',
                   help='Max colony diameter in px (default: 50)')
    p.add_argument('--min-isolation', type=float, default=0.0, metavar='MM',
                   help='Min isolation from neighbours in mm (default: 0.0)')
    p.add_argument('--min-circularity', type=float, default=0.0, metavar='0-1',
                   help='Min circularity 0–1 (default: 0.0)')
    p.add_argument('--exclude', type=str, default='', metavar='WELLS',
                   help='Comma-separated wells to exclude e.g. "A1,H11-H12"')
    p.add_argument('--annotated-image', action='store_true',
                   help='Also save an annotated PNG')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Print progress to stdout')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Resolve image path ──────────────────────────────────────────────────
    if args.run_folder:
        if not args.run_folder.is_dir():
            print(f"ERROR: run-folder not found: {args.run_folder}", file=sys.stderr)
            sys.exit(1)
        args.image = find_newest_image(args.run_folder)
        if args.image is None:
            print(f"ERROR: No image files found in {args.run_folder}", file=sys.stderr)
            sys.exit(1)
        if args.verbose:
            print(f"Auto-selected image: {args.image}")
    elif args.image is None:
        parser.print_help()
        sys.exit(1)

    if not args.image.is_file():
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # ── Output directory ────────────────────────────────────────────────────
    out_dir = args.output if args.output else args.image.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.image.stem   # e.g. "Original"

    if args.verbose:
        print(f"Input : {args.image}")
        print(f"Output: {out_dir}")

    # ── Load image ──────────────────────────────────────────────────────────
    if args.verbose:
        print("Loading image…")
    try:
        img_pil   = Image.open(args.image).convert('RGB')
        img_array = np.array(img_pil)
    except Exception as e:
        print(f"ERROR: Cannot open image: {e}", file=sys.stderr)
        sys.exit(1)

    H, W = img_array.shape[:2]
    if args.verbose:
        print(f"  Size: {W}×{H} px")

    # ── Calibrate ───────────────────────────────────────────────────────────
    if args.verbose:
        print("Calibrating plate…")
    gray = (img_array[:,:,0]*0.299 +
            img_array[:,:,1]*0.587 +
            img_array[:,:,2]*0.114).astype(np.float32)
    plate_cx, plate_cy, px_per_mm, bounds = detect_plate_calibration(gray)
    if args.verbose:
        print(f"  Centre: ({plate_cx:.1f}, {plate_cy:.1f}) px")
        print(f"  pxPerMm: {px_per_mm:.4f}")
        print(f"  Crop: {bounds}")

    # ── Detect ──────────────────────────────────────────────────────────────
    if args.verbose:
        print("Detecting colonies…")
    all_blobs = detect_colonies(
        img_array, plate_cx, plate_cy, px_per_mm, bounds,
        sensitivity=args.sensitivity,
        min_diam_px=args.min_diameter,
        max_diam_px=args.max_diameter,
        verbose=args.verbose,
    )
    if args.verbose:
        print(f"  Detected: {len(all_blobs)} candidates")

    # ── Select ──────────────────────────────────────────────────────────────
    exclude_wells = parse_exclusions(args.exclude)
    selected = select_colonies(
        all_blobs,
        target=args.target,
        target_format=args.target_format,
        min_isolation=args.min_isolation,
        min_circularity=args.min_circularity,
        min_radius_px=4.0,
        max_radius_px=50.0,
        exclude_wells=exclude_wells,
    )
    if args.verbose:
        print(f"  Selected: {len(selected)} colonies")

    if len(selected) == 0:
        print("WARNING: No colonies selected. Check image or try --sensitivity 7",
              file=sys.stderr)

    # ── Export CSV ──────────────────────────────────────────────────────────
    csv_path = out_dir / f"{args.target_name}_RearrayFile.csv"
    export_csv(
        selected, plate_cx, plate_cy, px_per_mm,
        args.source_name, args.target_name, args.target_format,
        csv_path,
    )
    if args.verbose:
        print(f"  Rearray CSV → {csv_path}")

    # ── Export summary JSON ─────────────────────────────────────────────────
    json_path = out_dir / f"{args.source_name}_summary.json"
    export_summary_json(
        all_blobs, selected, plate_cx, plate_cy, px_per_mm, bounds, args, json_path
    )
    if args.verbose:
        print(f"  Summary JSON → {json_path}")

    # ── Optional annotated image ────────────────────────────────────────────
    if args.annotated_image:
        ann_path = out_dir / f"{args.source_name}_annotated.png"
        export_annotated_image(img_array, all_blobs, selected, bounds, ann_path)
        if args.verbose:
            print(f"  Annotated PNG → {ann_path}")

    # ── Final summary to stdout (always) ───────────────────────────────────
    print(f"OK  {len(selected)}/{args.target} colonies selected  →  {csv_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
