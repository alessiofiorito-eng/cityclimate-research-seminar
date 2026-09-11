"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: tools/hsv_logger.py — HSV Diagnostic Logger.
Records per-cell HSV values with optional ground-truth labeling, to
help calibrate/validate the CV color-detection pipeline's HSV ranges.

Usage:
    cd cityclimate
    python tools/hsv_logger.py --realsense       # RealSense RGB (recommended)
    python tools/hsv_logger.py --camera          # regular webcam
    python tools/hsv_logger.py --demo            # synthetic frame
    python tools/hsv_logger.py --realsense --duration 60

Controls in the OpenCV window:
    Left click on a cell  → select that cell (highlighted in yellow)
    Color keys for the selected cell:
        r = red         g = darkgreen   G = lightgreen
        b = blue        w = white       W = lightgray
        d = darkgray    e = empty
    s = save a snapshot (JPG) now
    q = quit + save CSV

Output (in logs/):
    hsv_log_TIMESTAMP.csv    — per frame: cell, H/S/V median, detected color, ground truth
    snapshot_TIMESTAMP_N.jpg — manual snapshots
    summary_TIMESTAMP.txt    — per-color summary + suggested new ranges
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
TOOL_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOL_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from vision.cv_live_stream import (
    COLOR_RANGES,
    GRID_SIZE,
    _load_homography,
    _board_size,
    _center_crop,
    detect_cell_color,
    get_cell_hsv_median,
    make_screen_simulation_frame,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

KEY_TO_COLOR: dict[int, str] = {
    ord("r"): "red",
    ord("g"): "darkgreen",
    ord("G"): "lightgreen",
    ord("b"): "blue",
    ord("w"): "white",
    ord("W"): "lightgray",
    ord("d"): "darkgray",
    ord("e"): "empty",
}

COLOR_BGR_DISPLAY: dict[str, tuple[int, int, int]] = {
    "red":        (50,  50,  220),
    "darkgreen":  (0,   120,  0),
    "lightgreen": (50,  220,  50),
    "blue":       (220, 100,  50),
    "white":      (240, 240, 240),
    "lightgray":  (180, 180, 180),
    "darkgray":   (90,  90,  90),
    "empty":      (40,  40,  40),
}

DISPLAY_SIZE = 600


# ---------------------------------------------------------------------------
# Frame source abstraction
# ---------------------------------------------------------------------------

class _FrameSource:
    """Unified frame provider: RealSense / cv2 / demo."""

    def __init__(self, mode: str, camera_index: int = 0) -> None:
        """Initialize the requested frame source, falling back to demo
        mode if the requested hardware source is unavailable.

        Args:
            mode: One of "realsense", "camera", or "demo".
            camera_index: cv2.VideoCapture device index (only used with
                mode="camera").
        """
        self._mode = mode  # 'realsense' | 'camera' | 'demo'
        self._rs_cam = None
        self._cap: Optional[cv2.VideoCapture] = None

        if mode == "realsense":
            from camera.realsense_camera import RealSenseCamera
            print("[hsv_logger] Starte RealSense (Warmup ~3s)...")
            self._rs_cam = RealSenseCamera()
            if not self._rs_cam.is_open():
                print("[hsv_logger] RealSense nicht verfügbar — Demo-Modus.")
                self._mode = "demo"
        elif mode == "camera":
            from vision.cv_live_stream import _open_capture
            self._cap = _open_capture(camera_index) or cv2.VideoCapture(camera_index)
            if not self._cap.isOpened():
                print(f"[hsv_logger] Kamera {camera_index} nicht verfügbar — Demo-Modus.")
                self._mode = "demo"

    def read(self) -> Optional[np.ndarray]:
        """Return the next available frame from the active source."""
        if self._mode == "realsense":
            return self._rs_cam.get_frame()
        elif self._mode == "camera":
            ret, frame = self._cap.read()
            return frame if ret else None
        else:
            return make_screen_simulation_frame()

    def release(self) -> None:
        """Release the underlying camera resource, if any."""
        if self._rs_cam:
            self._rs_cam.release()
        if self._cap:
            self._cap.release()


# ---------------------------------------------------------------------------
# Helper: warp frame to board
# ---------------------------------------------------------------------------

def _warp_frame(frame: np.ndarray) -> np.ndarray:
    """Warp a raw frame to the canonical board view using the saved
    homography, falling back to a simple center crop if no calibration
    is available or the warp fails."""
    H = _load_homography()
    if H is not None:
        try:
            w, h = _board_size()
            return cv2.warpPerspective(frame, H, (w, h))
        except Exception:
            pass
    return _center_crop(frame)


# ---------------------------------------------------------------------------
# Draw annotated board
# ---------------------------------------------------------------------------

def _draw_board(
    board: np.ndarray,
    detected: list[list[str]],
    ground_truth: dict[tuple[int, int], str],
    selected_cell: Optional[tuple[int, int]],
    display_size: int = DISPLAY_SIZE,
) -> np.ndarray:
    """Render the board with per-cell grid lines, detected-color labels,
    ground-truth labels (if set), and match/mismatch/selection border
    coloring.

    Border colors: gray = no ground truth yet, green = detected matches
    ground truth, red = mismatch. The currently selected cell is
    highlighted with a thick yellow border regardless of match state.

    Args:
        board: The warped board frame (BGR) to annotate.
        detected: The current GRID_SIZE x GRID_SIZE detected color
            matrix.
        ground_truth: Mapping of (row, col) to the manually labeled
            ground-truth color name.
        selected_cell: The currently selected (row, col), or None.
        display_size: Output image size in pixels (square).

    Returns:
        The resized and annotated board image.
    """
    disp = cv2.resize(board, (display_size, display_size), interpolation=cv2.INTER_LINEAR)
    cs   = display_size // GRID_SIZE

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x1, y1 = col * cs, row * cs
            x2, y2 = x1 + cs, y1 + cs
            det    = detected[row][col] if detected else "empty"
            truth  = ground_truth.get((row, col))

            if truth is None:
                border, thick = (160, 160, 160), 1
            elif det == truth:
                border, thick = (0, 220, 80), 2
            else:
                border, thick = (0, 60, 220), 2

            if selected_cell == (row, col):
                cv2.rectangle(disp, (x1, y1), (x2-1, y2-1), (0, 255, 255), 3)
            else:
                cv2.rectangle(disp, (x1, y1), (x2-1, y2-1), border, thick)

            cv2.putText(disp, det[:6], (x1+3, y1+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1, cv2.LINE_AA)
            if truth:
                gt_bgr = COLOR_BGR_DISPLAY.get(truth, (200,200,200))
                cv2.putText(disp, f">{truth[:6]}", (x1+3, y2-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, gt_bgr, 1, cv2.LINE_AA)

    for i in range(1, GRID_SIZE):
        cv2.line(disp, (i*cs, 0), (i*cs, display_size), (60,60,60), 1)
        cv2.line(disp, (0, i*cs), (display_size, i*cs), (60,60,60), 1)
    return disp


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _draw_sidebar(
    height: int,
    ground_truth: dict[tuple[int, int], str],
    detected: list[list[str]],
    frame_count: int,
    elapsed: float,
) -> np.ndarray:
    """Render the info sidebar: control legend, frame/time/label
    counters, and a live per-color detection accuracy breakdown
    computed from the current ground-truth labels.

    Args:
        height: Sidebar image height in pixels (matches the board
            display height).
        ground_truth: Mapping of (row, col) to manually labeled color.
        detected: The current GRID_SIZE x GRID_SIZE detected color
            matrix.
        frame_count: Number of frames processed so far.
        elapsed: Elapsed session time in seconds.

    Returns:
        The rendered sidebar image (fixed width, `height` tall).
    """
    sidebar = np.full((height, 260, 3), 20, dtype=np.uint8)
    lines = [
        "== HSV Logger ==", "",
        "Klick = Zelle waehlen",
        "r=red  g=dkgrn  G=ltgrn",
        "b=blue w=white  W=ltgry",
        "d=dkgry  e=empty",
        "s=Snapshot  q=Beenden",
        "",
        f"Frames: {frame_count}",
        f"Zeit:   {elapsed:.1f}s",
        f"Labels: {len(ground_truth)}/25",
        "", "--- Genauigkeit ---",
    ]
    per_color: dict[str, list[int]] = {}
    for (r, c), truth in ground_truth.items():
        det = detected[r][c] if detected else "empty"
        if truth not in per_color:
            per_color[truth] = [0, 0]
        per_color[truth][1] += 1
        if det == truth:
            per_color[truth][0] += 1
    for color, (correct, total) in sorted(per_color.items()):
        pct = int(100 * correct / total) if total else 0
        lines.append(f"{color[:9]:<9}: {pct:3d}%")
    y = 18
    for line in lines:
        clr = (0, 220, 180) if line.startswith("==") else (180, 180, 180)
        cv2.putText(sidebar, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, clr, 1, cv2.LINE_AA)
        y += 17
    return sidebar


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_logger(
    mode: str = "demo",
    camera_index: int = 0,
    duration: Optional[float] = None,
) -> None:
    """Run the interactive HSV diagnostic logger session.

    Opens the chosen frame source and an OpenCV display window, then
    loops: reading a frame, warping it to the board, running per-cell
    color detection, logging per-cell HSV medians and any ground-truth
    labels, and rendering the annotated board + info sidebar. Handles
    mouse clicks (cell selection), color-label key presses, manual
    snapshots, and quitting. On exit, writes the accumulated log rows
    to CSV, writes a per-color summary report, and saves a final
    snapshot.

    Args:
        mode: One of "realsense", "camera", or "demo".
        camera_index: cv2.VideoCapture device index (mode="camera" only).
        duration: Optional automatic stop time in seconds.
    """
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path     = LOG_DIR / f"hsv_log_{timestamp}.csv"
    summary_path = LOG_DIR / f"summary_{timestamp}.txt"

    ground_truth:  dict[tuple[int, int], str] = {}
    selected_cell: Optional[tuple[int, int]]  = None
    log_rows:      list[dict] = []
    snapshot_count = 0
    frame_count    = 0
    start_time     = time.time()

    source = _FrameSource(mode, camera_index)

    win = "HSV Logger - q=Beenden"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISPLAY_SIZE + 260, DISPLAY_SIZE)

    def _on_mouse(event, x, y, flags, param):
        nonlocal selected_cell
        if event == cv2.EVENT_LBUTTONDOWN and x < DISPLAY_SIZE:
            cs  = DISPLAY_SIZE // GRID_SIZE
            col = min(GRID_SIZE - 1, x // cs)
            row = min(GRID_SIZE - 1, y // cs)
            selected_cell = (row, col)
            print(f"[select] Zelle ({row+1},{col+1}) — jetzt Farbtaste druecken")

    cv2.setMouseCallback(win, _on_mouse)
    print("\n[hsv_logger] Gestartet. Klick auf Zelle dann Farbtaste.")
    print("'s'=Snapshot  'q'=Beenden & speichern\n")

    last_board:    Optional[np.ndarray] = None
    last_detected: list[list[str]]      = [["empty"] * GRID_SIZE for _ in range(GRID_SIZE)]

    while True:
        elapsed = time.time() - start_time
        if duration and elapsed >= duration:
            print(f"[hsv_logger] {duration}s erreicht — speichere...")
            break

        raw = source.read()
        if raw is None:
            time.sleep(0.03)
            continue

        board      = _warp_frame(raw)
        last_board = board

        h_roi, w_roi = board.shape[:2]
        cs_x = w_roi // GRID_SIZE
        cs_y = h_roi // GRID_SIZE
        hsv_board = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)

        detected = []
        row_data: list[dict] = []
        for row in range(GRID_SIZE):
            row_det = []
            for col in range(GRID_SIZE):
                x1, y1   = col * cs_x, row * cs_y
                cell_hsv = hsv_board[y1:y1+cs_y, x1:x1+cs_x]
                det      = detect_cell_color(cell_hsv) if cell_hsv.size > 0 else "empty"
                med      = get_cell_hsv_median(board, row, col, cell_size=cs_x)
                truth    = ground_truth.get((row, col), "")
                row_det.append(det)
                row_data.append({
                    "timestamp": f"{elapsed:.3f}",
                    "frame":     frame_count,
                    "row":       row,
                    "col":       col,
                    "h_med":     med[0] if med else "",
                    "s_med":     med[1] if med else "",
                    "v_med":     med[2] if med else "",
                    "detected":  det,
                    "truth":     truth,
                    "match":     ("1" if det == truth else "0") if truth else "",
                })
            detected.append(row_det)
        last_detected = detected
        log_rows.extend(row_data)
        frame_count += 1

        board_disp = _draw_board(board, detected, ground_truth, selected_cell)
        sidebar    = _draw_sidebar(DISPLAY_SIZE, ground_truth, detected, frame_count, elapsed)
        cv2.imshow(win, np.hstack([board_disp, sidebar]))

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            if last_board is not None:
                p = LOG_DIR / f"snapshot_{timestamp}_{snapshot_count}.jpg"
                cv2.imwrite(str(p), last_board)
                print(f"[snapshot] {p}")
                snapshot_count += 1
        elif key in KEY_TO_COLOR:
            if selected_cell is not None:
                color = KEY_TO_COLOR[key]
                ground_truth[selected_cell] = color
                r, c = selected_cell
                print(f"[label] ({r+1},{c+1}) = {color}")
            else:
                print("[label] Erst Zelle anklicken!")

    cv2.destroyAllWindows()
    source.release()

    if log_rows:
        fieldnames = ["timestamp","frame","row","col",
                      "h_med","s_med","v_med","detected","truth","match"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"[hsv_logger] CSV: {csv_path}  ({len(log_rows)} Zeilen)")

    _write_summary(summary_path, log_rows, ground_truth)

    if last_board is not None:
        p = LOG_DIR / f"snapshot_{timestamp}_final.jpg"
        cv2.imwrite(str(p), last_board)
        print(f"[hsv_logger] Final-Snapshot: {p}")

    print(f"[hsv_logger] Fertig. Logs in: {LOG_DIR}")


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _write_summary(
    path: Path,
    log_rows: list[dict],
    ground_truth: dict[tuple[int, int], str],
) -> None:
    """Compute and write a per-color summary report: label/accuracy
    counts, HSV median/std/min/max statistics per labeled color, and
    suggested new HSV ranges (median ± std + fixed margin) that can be
    copied into color_ranges.json.

    Args:
        path: Output path for the summary text file.
        log_rows: All accumulated per-cell log row dicts from the
            session.
        ground_truth: Mapping of (row, col) to manually labeled color,
            used only for the labeled-cell count in the header.
    """
    color_hsv:     dict[str, list] = {}
    color_correct: dict[str, int]  = {}
    color_total:   dict[str, int]  = {}

    for entry in log_rows:
        truth = entry.get("truth", "")
        if not truth:
            continue
        h, s, v = entry.get("h_med"), entry.get("s_med"), entry.get("v_med")
        if h == "" or s == "" or v == "":
            continue
        color_hsv.setdefault(truth, [])
        color_correct.setdefault(truth, 0)
        color_total.setdefault(truth, 0)
        color_hsv[truth].append((int(h), int(s), int(v)))
        color_total[truth] += 1
        if entry.get("match") == "1":
            color_correct[truth] += 1

    lines = [
        "HSV Logger — Zusammenfassung",
        "=" * 50,
        f"Labeled cells : {len(ground_truth)}/25",
        f"Total log rows: {len(log_rows)}",
        "",
        f"{'Farbe':<14} {'Acc%':>5}  {'H-Med':>6} {'S-Med':>6} {'V-Med':>6}"
        f"  {'H-Std':>6} {'S-Std':>6} {'V-Std':>6}  {'H-Min':>6} {'H-Max':>6}",
        "-" * 80,
    ]

    for color in sorted(color_hsv.keys()):
        vals  = np.array(color_hsv[color])
        h_m   = int(np.median(vals[:,0]))
        s_m   = int(np.median(vals[:,1]))
        v_m   = int(np.median(vals[:,2]))
        h_std = int(np.std(vals[:,0]))
        s_std = int(np.std(vals[:,1]))
        v_std = int(np.std(vals[:,2]))
        acc   = int(100 * color_correct[color] / color_total[color]) if color_total[color] else 0
        lines.append(
            f"{color:<14} {acc:>5}%  {h_m:>6} {s_m:>6} {v_m:>6}"
            f"  {h_std:>6} {s_std:>6} {v_std:>6}"
            f"  {int(vals[:,0].min()):>6} {int(vals[:,0].max()):>6}"
        )

    lines += [
        "",
        "Tipp: H-Std > 15 = breite Hue-Streuung → Range ausweiten",
        "      Acc% < 80  = HSV-Range passt nicht → anpassen",
        "",
        "Suggested ranges (Median +/- Std + Margin):",
        "-" * 80,
    ]

    MARGIN_H, MARGIN_S, MARGIN_V = 10, 30, 30
    for color in sorted(color_hsv.keys()):
        vals  = np.array(color_hsv[color])
        h_m   = int(np.median(vals[:,0]))
        s_m   = int(np.median(vals[:,1]))
        v_m   = int(np.median(vals[:,2]))
        h_std = int(np.std(vals[:,0]))
        s_std = int(np.std(vals[:,1]))
        v_std = int(np.std(vals[:,2]))
        lines.append(
            f"  {color:<14}: lower=[{max(0,h_m-h_std-MARGIN_H):3d}, "
            f"{max(0,s_m-s_std-MARGIN_S):3d}, {max(0,v_m-v_std-MARGIN_V):3d}]  "
            f"upper=[{min(179,h_m+h_std+MARGIN_H):3d}, 255, 255]"
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[hsv_logger] Summary: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSV Diagnostic Logger")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--realsense", action="store_true", help="RealSense RGB via pyrealsense2 (empfohlen)")
    group.add_argument("--camera",    action="store_true", help="Normale Webcam via cv2")
    group.add_argument("--demo",      action="store_true", help="Demo-Modus (synthetische Frames)")
    parser.add_argument("--index",    type=int,   default=0,    help="Kamera-Index (nur mit --camera)")
    parser.add_argument("--duration", type=float, default=None, help="Automatisch nach N Sekunden stoppen")
    args = parser.parse_args()

    if args.realsense:
        mode = "realsense"
    elif args.camera:
        mode = "camera"
    else:
        mode = "demo"

    run_logger(mode=mode, camera_index=args.index, duration=args.duration)
