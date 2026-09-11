"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: vision/cv_live_stream.py — Live computer-vision detection pipeline
for the CityClimate Board. This is the single authoritative CV pipeline
used by the GUI; do NOT use color_classifier.py or grid_extractor.py for
live detection — both are kept only as legacy stubs.

Provides: homography-based board warping, HSV color-range based per-cell
color classification, temporal stabilization via majority voting across
frames, a synthetic "screen simulation" frame generator for demo mode,
and a background-thread LiveStreamDetector that continuously reads
frames (from a real camera, a shared frame provider, or demo mode) and
reports a stable color matrix once it settles.
"""

import cv2
import json
import numpy as np
import platform
from pathlib import Path
from typing import Optional, Callable
import threading
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FRAME_SIZE = 480
GRID_SIZE  = 5
CELL_SIZE  = 96

# Inner crop: ignore outer 15% of each cell edge (grid lines + border shadows)
CELL_MARGIN = 0.15

STABLE_WINDOW    = 15
STABLE_THRESHOLD = 10

_PREFERRED_BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
_FALLBACK_BACKEND  = cv2.CAP_MSMF  if platform.system() == "Windows" else cv2.CAP_ANY
_OPEN_TIMEOUT_MS   = 2000
_MAX_READ_FAILURES = 10

_CALIBRATION_FILE  = Path("calibration.json")
_COLOR_RANGES_FILE = Path("color_ranges.json")


def _board_size() -> tuple[int, int]:
    """Return the canonical (width, height) in pixels the board should be
    warped to, read from config.py, falling back to FRAME_SIZE if config
    cannot be imported."""
    try:
        import config
        return int(config.BOARD_WIDTH_PX), int(config.BOARD_HEIGHT_PX)
    except Exception:
        return FRAME_SIZE, FRAME_SIZE


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------

def _load_homography() -> Optional[np.ndarray]:
    """Load the saved homography matrix from the calibration file, if any.

    Returns:
        A 3x3 float32 numpy array, or None if no calibration file exists
        or it could not be parsed.
    """
    if not _CALIBRATION_FILE.exists():
        return None
    try:
        data = json.loads(_CALIBRATION_FILE.read_text(encoding="utf-8"))
        H = data.get("homography_matrix")
        if H is not None:
            return np.array(H, dtype=np.float32)
    except Exception as exc:
        print(f"[cv_live_stream] Warning: could not load calibration: {exc}")
    return None


def apply_homography_warp(frame: np.ndarray) -> Optional[np.ndarray]:
    """Warp a raw camera frame onto the canonical board rectangle using the
    saved homography, if calibration data is available.

    Args:
        frame: Raw BGR frame from the camera.

    Returns:
        The perspective-warped frame, or None if no calibration is
        available or the warp fails.
    """
    H = _load_homography()
    if H is None:
        return None
    try:
        w, h = _board_size()
        return cv2.warpPerspective(frame, H, (w, h))
    except Exception as exc:
        print(f"[cv_live_stream] Warp failed: {exc}")
        return None


def _center_crop(frame: np.ndarray) -> np.ndarray:
    """Fallback ROI extraction: center-crop the frame to FRAME_SIZE x
    FRAME_SIZE (used when no homography calibration is available)."""
    h, w = frame.shape[:2]
    xf = max(0, (w - FRAME_SIZE) // 2)
    yf = max(0, (h - FRAME_SIZE) // 2)
    roi = frame[yf:yf + FRAME_SIZE, xf:xf + FRAME_SIZE]
    if roi.shape[0] < FRAME_SIZE or roi.shape[1] < FRAME_SIZE:
        return cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
    return roi.copy()


def _get_roi(frame: np.ndarray) -> np.ndarray:
    """Return the region of interest for detection: the homography-warped
    board if calibrated, otherwise a simple center crop."""
    warped = apply_homography_warp(frame)
    if warped is not None:
        return warped
    return _center_crop(frame)


# ---------------------------------------------------------------------------
# HSV color ranges
# VALIDATED DEFAULTS — computed from median HSV values in a logged
# calibration session. See paper for details on the calibration procedure.
# Margins: H±10, S±35, V±35 (based on actual log variance)
# ---------------------------------------------------------------------------
_VALIDATED_DEFAULTS: dict[str, list[list[int]]] = {
    # darkgreen: H=64, S=138, V=156
    "darkgreen":  [[ 53,  80, 110], [ 77, 210, 200]],
    # darkgray:   H=9,  S=37,  V=97
    "darkgray":   [[  0,   0,  50], [ 20,  80, 160]],
    # white:      H=11, S=34,  V=246
    "white":      [[  0,   0, 200], [ 25,  70, 255]],
    # red:        H=174, S=157, V=115 — wraps around, two ranges
    "red":        [[165, 100,  75], [179, 200, 160]],
    "red2":       [[  0, 100,  75], [ 12, 200, 160]],
    # blue:       H=104, S=217, V=177
    "blue":       [[ 94, 140, 100], [116, 255, 255]],
    # lightgreen and lightgray kept from previous validated run
    "lightgreen": [[ 51, 180, 170], [ 90, 255, 255]],
    "lightgray":  [[  0,   0, 155], [180,  50, 225]],
}

COLOR_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

# Colors currently disabled by the user (loaded from color_ranges.json _disabled key)
DISABLED_COLORS: set[str] = set()


def _build_ranges_from_dict(d: dict[str, list[list[int]]]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Convert a flat {color_name: [[lo], [hi]]} dict into the internal
    COLOR_RANGES representation, merging the "red2" wrap-around range into
    the "red" entry as a second (lo, hi) pair.

    Args:
        d: Flat dict mapping color name (or "red2") to [lo_hsv, hi_hsv].

    Returns:
        A dict mapping color name to a list of (lo, hi) numpy array pairs.
    """
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for name, pair in d.items():
        if name == "red2":
            continue
        lo = np.array(pair[0], dtype=np.uint8)
        hi = np.array(pair[1], dtype=np.uint8)
        out[name] = [(lo, hi)]
    if "red2" in d:
        lo2 = np.array(d["red2"][0], dtype=np.uint8)
        hi2 = np.array(d["red2"][1], dtype=np.uint8)
        out.setdefault("red", [])
        out["red"] = [out["red"][0], (lo2, hi2)]
    return out


def _load_color_ranges_from_file() -> None:
    """Load color_ranges.json into COLOR_RANGES.
    Falls back to _VALIDATED_DEFAULTS for any missing color.
    Keys starting with '_' (e.g. '_disabled') are treated as metadata and skipped.
    """
    global DISABLED_COLORS
    merged = dict(_VALIDATED_DEFAULTS)
    disabled: list[str] = []

    if _COLOR_RANGES_FILE.exists():
        try:
            data = json.loads(_COLOR_RANGES_FILE.read_text(encoding="utf-8"))
            for name, value in data.items():
                if name.startswith("_"):
                    # Metadata key — handle known ones, skip rest
                    if name == "_disabled" and isinstance(value, list):
                        disabled = [str(v) for v in value]
                    continue
                merged[name] = value
        except Exception as exc:
            print(f"[cv_live_stream] Warning: could not load {_COLOR_RANGES_FILE}: {exc}")

    DISABLED_COLORS = set(disabled)

    # Remove disabled colors from the merged dict before building ranges
    for color in disabled:
        merged.pop(color, None)
        merged.pop(color + "2", None)  # also remove red2 if red is disabled

    COLOR_RANGES.clear()
    COLOR_RANGES.update(_build_ranges_from_dict(merged))

    if disabled:
        print(f"[cv_live_stream] Disabled colors (treated as empty): {disabled}")


def save_color_ranges_to_file(
    ranges: Optional[dict[str, list[tuple[np.ndarray, np.ndarray]]]] = None,
) -> None:
    """Persist COLOR_RANGES (or supplied dict) to color_ranges.json.
    Preserves the _disabled metadata key if already present in the file.
    """
    src = ranges if ranges is not None else COLOR_RANGES
    out: dict = {}

    # Preserve existing metadata keys (e.g. _disabled)
    if _COLOR_RANGES_FILE.exists():
        try:
            existing = json.loads(_COLOR_RANGES_FILE.read_text(encoding="utf-8"))
            for k, v in existing.items():
                if k.startswith("_"):
                    out[k] = v
        except Exception:
            pass

    for name, pairs in src.items():
        if pairs:
            lo, hi = pairs[0]
            out[name] = [lo.tolist(), hi.tolist()]
            if name == "red" and len(pairs) > 1:
                lo2, hi2 = pairs[1]
                out["red2"] = [lo2.tolist(), hi2.tolist()]

    _COLOR_RANGES_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")


# Populate COLOR_RANGES at import time from disk (or validated defaults)
_load_color_ranges_from_file()


# ---------------------------------------------------------------------------
# LCZ mapping — MUST stay in sync with config.COLOR_TO_LCZ
# ---------------------------------------------------------------------------
COLOR_TO_LCZ: dict[str, str | None] = {
    "red":        "2",
    "lightgray":  "5",
    "white":      "6",
    "darkgray":   "9",
    "darkgreen":  "A",
    "lightgreen": "D",
    "blue":       "G",
    "empty":      None,
}


# ---------------------------------------------------------------------------
# Core detection: inner-crop + pixel-fraction voting
# ---------------------------------------------------------------------------

def _inner_crop(cell_hsv: np.ndarray) -> np.ndarray:
    """Strip outer CELL_MARGIN fraction from each edge."""
    h, w = cell_hsv.shape[:2]
    m_y  = max(1, int(h * CELL_MARGIN))
    m_x  = max(1, int(w * CELL_MARGIN))
    return cell_hsv[m_y:h - m_y, m_x:w - m_x]


def detect_cell_color(cell_hsv: np.ndarray) -> str:
    """Return best color name for one cell via pixel-fraction voting.

    Crops the cell's inner region (to avoid grid-line/shadow noise), then
    for each candidate color computes the fraction of pixels falling
    inside its HSV range(s). The color with the highest fraction above a
    minimum threshold (0.25) wins; otherwise the cell is reported empty.
    """
    inner = _inner_crop(cell_hsv)
    if inner.size == 0:
        return "empty"

    flat          = inner.reshape(-1, 3)
    n_pixels      = flat.shape[0]
    best_color    = "empty"
    best_fraction = 0.25

    for color, ranges in COLOR_RANGES.items():
        mask = np.zeros(n_pixels, dtype=bool)
        for lo, hi in ranges:
            # A pixel matches if it falls within ANY of this color's HSV
            # range pairs (relevant for "red", which wraps around H=0/179).
            mask |= np.all((flat >= lo) & (flat <= hi), axis=1)
        fraction = mask.sum() / n_pixels
        if fraction > best_fraction:
            best_fraction = fraction
            best_color    = color

    return best_color


def detect_matrix(
    frame_roi: np.ndarray,
    cell_size: Optional[int] = None,
) -> tuple[list[list[str]], bool]:
    """Slice the board ROI into a GRID_SIZE x GRID_SIZE grid and classify
    each cell's color.

    Args:
        frame_roi: The warped/cropped board region of interest (BGR).
        cell_size: Optional fixed cell size in pixels; if omitted, it is
            derived by dividing the ROI dimensions by GRID_SIZE.

    Returns:
        A tuple (matrix, valid) where matrix is a GRID_SIZE x GRID_SIZE
        list of color name strings, and valid is True only if every cell
        was classified as something other than "empty".
    """
    h_roi, w_roi = frame_roi.shape[:2]
    cs_x = cell_size if cell_size else w_roi // GRID_SIZE
    cs_y = cell_size if cell_size else h_roi // GRID_SIZE
    hsv  = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
    matrix = []
    for row in range(GRID_SIZE):
        row_result = []
        for col in range(GRID_SIZE):
            x1, y1   = col * cs_x, row * cs_y
            cell_hsv = hsv[y1:y1 + cs_y, x1:x1 + cs_x]
            if cell_hsv.size == 0:
                row_result.append("empty")
                continue
            row_result.append(detect_cell_color(cell_hsv))
        matrix.append(row_result)
    valid = all(c != "empty" for row in matrix for c in row)
    return matrix, valid


def get_cell_hsv_median(
    frame_roi: np.ndarray,
    row: int,
    col: int,
    cell_size: Optional[int] = None,
) -> Optional[tuple[int, int, int]]:
    """Compute the median HSV value of a single cell's inner region.

    Used by calibration/logging tools (e.g. hsv_logger.py) to sample
    representative HSV values for a tile without running full color
    classification.

    Args:
        frame_roi: The warped/cropped board region of interest (BGR).
        row: Grid row index of the target cell.
        col: Grid column index of the target cell.
        cell_size: Optional fixed cell size in pixels; if omitted, it is
            derived by dividing the ROI dimensions by GRID_SIZE.

    Returns:
        A (h, s, v) tuple of median HSV values, or None if the cell could
        not be sampled (e.g. out of bounds or empty region).
    """
    try:
        h_roi, w_roi = frame_roi.shape[:2]
        cs_x = cell_size if cell_size else w_roi // GRID_SIZE
        cs_y = cell_size if cell_size else h_roi // GRID_SIZE
        hsv  = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
        x1, y1 = col * cs_x, row * cs_y
        cell   = hsv[y1:y1 + cs_y, x1:x1 + cs_x]
        if cell.size == 0:
            return None
        inner = _inner_crop(cell)
        flat  = inner.reshape(-1, 3)
        if flat.shape[0] == 0:
            return None
        med = np.median(flat, axis=0).astype(int)
        return int(med[0]), int(med[1]), int(med[2])
    except Exception:
        return None


def compute_ranges_from_samples(
    samples: dict[str, list[tuple[int, int, int]]],
    h_margin: int = 10,
    s_margin: int = 35,
    v_margin: int = 35,
) -> dict[str, list[list[int]]]:
    """Compute HSV ranges dynamically from labeled sample points.

    For each color, derives a bounding HSV range from the min/max of its
    sampled points, expanded by the given margins. Colors with no samples
    fall back to _VALIDATED_DEFAULTS.

    Args:
        samples: Mapping of color name to a list of (h, s, v) sample tuples.
        h_margin: Margin added/subtracted on the hue bounds.
        s_margin: Margin subtracted from the minimum saturation bound.
        v_margin: Margin subtracted from the minimum value bound.

    Returns:
        A dict mapping color name to [lo_hsv, hi_hsv] range lists,
        suitable for feeding into _build_ranges_from_dict / persisting to
        color_ranges.json.
    """
    result: dict[str, list[list[int]]] = {}
    for color, pts in samples.items():
        if not pts:
            if color in _VALIDATED_DEFAULTS:
                result[color] = _VALIDATED_DEFAULTS[color]
            continue
        h_vals = [p[0] for p in pts]
        s_vals = [p[1] for p in pts]
        v_vals = [p[2] for p in pts]
        h_lo = max(0,   int(min(h_vals)) - h_margin)
        h_hi = min(179, int(max(h_vals)) + h_margin)
        s_lo = max(0,   int(min(s_vals)) - s_margin)
        v_lo = max(0,   int(min(v_vals)) - v_margin)
        result[color] = [[h_lo, s_lo, v_lo], [h_hi, 255, 255]]
    for color, default in _VALIDATED_DEFAULTS.items():
        if color not in result:
            result[color] = default
    return result


def draw_debug(frame_roi: np.ndarray, matrix: list[list[str]]) -> None:
    """Draw grid lines and detected color labels onto frame_roi in place,
    for visual debugging of the detection pipeline."""
    h_roi, w_roi = frame_roi.shape[:2]
    cs_x = w_roi // GRID_SIZE
    cs_y = h_roi // GRID_SIZE
    for i in range(1, GRID_SIZE):
        cv2.line(frame_roi, (i * cs_x, 0),  (i * cs_x, h_roi), (100, 100, 100), 1)
        cv2.line(frame_roi, (0, i * cs_y),  (w_roi, i * cs_y), (100, 100, 100), 1)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            color = matrix[row][col]
            tc    = (0, 255, 0) if color != "empty" else (0, 0, 255)
            cv2.putText(
                frame_roi, color,
                (col * cs_x + 3, row * cs_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, tc, 1,
            )


def color_matrix_to_lcz(
    matrix: list[list[str]],
) -> list[list[str | None]]:
    """Translate a detected color-name matrix into an LCZ-ID matrix using
    COLOR_TO_LCZ."""
    return [[COLOR_TO_LCZ.get(cell) for cell in row] for row in matrix]


# ---------------------------------------------------------------------------
# StableMatrix: majority-vote over last STABLE_WINDOW frames per cell
# ---------------------------------------------------------------------------

class StableMatrix:
    """Temporal stabilizer for the detected color grid.

    Maintains a per-cell ring buffer of the last `window` detected colors
    and reports, for each cell, the majority color if it appears at least
    `threshold` times within that window. This smooths out flicker/noise
    in the raw per-frame detection before it is surfaced to the rest of
    the application.
    """

    def __init__(
        self,
        rows: int = GRID_SIZE,
        cols: int = GRID_SIZE,
        window: int = STABLE_WINDOW,
        threshold: int = STABLE_THRESHOLD,
    ) -> None:
        """Initialize per-cell ring buffers.

        Args:
            rows: Number of grid rows to track.
            cols: Number of grid columns to track.
            window: Number of recent frames to keep per cell.
            threshold: Minimum vote count within the window required for a
                color to be considered "stable".
        """
        self._rows      = rows
        self._cols      = cols
        self._window    = window
        self._threshold = threshold
        self._buffers: list[list[list[str]]] = [
            [["empty"] * window for _ in range(cols)] for _ in range(rows)
        ]
        self._pos:    list[list[int]] = [[0] * cols for _ in range(rows)]
        self._stable: list[list[str]] = [["empty"] * cols for _ in range(rows)]

    def update(self, raw_matrix: list[list[str]]) -> list[list[str]]:
        """Feed one new raw detection matrix into the ring buffers and
        return the current stable matrix.

        Args:
            raw_matrix: The latest per-frame detected color matrix.

        Returns:
            A copy of the current stabilized color matrix.
        """
        for r in range(self._rows):
            for c in range(self._cols):
                color = (
                    raw_matrix[r][c]
                    if r < len(raw_matrix) and c < len(raw_matrix[r])
                    else "empty"
                )
                buf = self._buffers[r][c]
                pos = self._pos[r][c]
                buf[pos % self._window] = color
                self._pos[r][c] = (pos + 1) % self._window
                # Majority vote across the ring buffer for this cell
                counts: dict[str, int] = {}
                for v in buf:
                    counts[v] = counts.get(v, 0) + 1
                best = max(counts, key=lambda k: counts[k])
                if counts[best] >= self._threshold:
                    self._stable[r][c] = best
        return [row[:] for row in self._stable]

    def reset(self) -> None:
        """Clear all ring buffers and stable state back to "empty"."""
        for r in range(self._rows):
            for c in range(self._cols):
                self._buffers[r][c] = ["empty"] * self._window
                self._pos[r][c]     = 0
                self._stable[r][c]  = "empty"


# ---------------------------------------------------------------------------
# Synthetic demo frame
# ---------------------------------------------------------------------------

_DEMO_MATRIX: list[list[str]] = [
    ["darkgreen",  "lightgreen", "blue",        "lightgray",  "red"       ],
    ["lightgreen", "darkgreen",  "lightgreen",  "white",      "darkgray"  ],
    ["blue",       "lightgreen", "darkgray",    "lightgray",  "lightgreen"],
    ["lightgray",  "white",      "lightgreen",  "darkgreen",  "blue"      ],
    ["red",        "darkgray",   "lightgreen",  "blue",       "darkgreen" ],
]

_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "blue":       (200, 100,  50),
    "darkgray":   ( 80,  80,  80),
    "darkgreen":  (  0,  80,   0),
    "lightgray":  (180, 180, 180),
    "lightgreen": ( 50, 200,  50),
    "red":        ( 50,  50, 200),
    "white":      (240, 240, 240),
    "empty":      ( 40,  40,  40),
}


def make_screen_simulation_frame(
    matrix: list[list[str]] | None = None,
) -> np.ndarray:
    """Render a synthetic BGR frame representing a color matrix, for use
    in demo mode when no physical camera/board is available.

    Args:
        matrix: Optional color-name matrix to render; defaults to
            _DEMO_MATRIX if omitted.

    Returns:
        A FRAME_SIZE x FRAME_SIZE x 3 uint8 BGR image.
    """
    src   = matrix if matrix is not None else _DEMO_MATRIX
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cn  = src[row][col] if row < len(src) and col < len(src[row]) else "empty"
            bgr = _COLOR_BGR.get(cn, _COLOR_BGR["empty"])
            x1, y1 = col * CELL_SIZE, row * CELL_SIZE
            frame[y1:y1 + CELL_SIZE, x1:x1 + CELL_SIZE] = bgr
    return frame


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _open_capture(index: int) -> Optional[cv2.VideoCapture]:
    """Attempt to open a camera at `index`, trying the preferred backend
    first and falling back to a secondary backend on failure.

    Returns:
        An opened cv2.VideoCapture, or None if both backends failed.
    """
    for backend in (_PREFERRED_BACKEND, _FALLBACK_BACKEND):
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _OPEN_TIMEOUT_MS)
        if cap.open(index, backend) and cap.isOpened():
            return cap
        cap.release()
    return None


# ---------------------------------------------------------------------------
# Background-thread detector
# ---------------------------------------------------------------------------

class LiveStreamDetector:
    """Runs the CV detection pipeline continuously on a background thread.

    Supports three modes depending on the constructor arguments:
      - Demo mode: repeatedly renders and classifies a synthetic frame.
      - Shared frame provider: pulls frames from an externally supplied
        callable (e.g. a frame source shared with another consumer).
      - Own camera: opens and reads directly from a cv2.VideoCapture
        device, with automatic reconnect on repeated read failures.

    Once the stabilized matrix settles (all cells non-empty for
    STABLE_WINDOW consecutive frames), `on_matrix_ready` is invoked
    exactly once per detection cycle (until reset()).
    """

    def __init__(
        self,
        on_matrix_ready: Callable[[list[list[str]]], None],
        demo: bool = False,
        camera_index: int = 0,
        frame_provider: Optional[Callable[[], np.ndarray]] = None,
    ) -> None:
        """Configure the detector.

        Args:
            on_matrix_ready: Callback invoked with the stabilized color
                matrix once detection settles.
            demo: If True, run in synthetic demo mode instead of using a
                real camera.
            camera_index: Device index to use when opening an own camera.
            frame_provider: Optional callable returning the latest BGR
                frame from an externally managed camera/source.
        """
        self.on_matrix_ready = on_matrix_ready
        self.demo            = demo
        self.camera_index    = camera_index
        self._frame_provider = frame_provider
        self._thread: Optional[threading.Thread] = None
        self._running        = False
        self.last_frame:  Optional[np.ndarray]      = None
        self.last_matrix: Optional[list[list[str]]] = None
        self._stable_matrix  = StableMatrix()
        self._matrix_sent    = False

    def start(self) -> None:
        """Start the background detection thread."""
        self._running     = True
        self._matrix_sent = False
        self._stable_matrix.reset()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background detection thread to stop."""
        self._running = False

    def reset(self, clear_last_matrix: bool = False) -> None:
        """Reset stabilization state so a new matrix can be detected/sent.

        Args:
            clear_last_matrix: If True, also clear last_matrix back to an
                all-empty grid.
        """
        self._matrix_sent = False
        self._stable_matrix.reset()
        if clear_last_matrix:
            self.last_matrix = [["empty"] * GRID_SIZE for _ in range(GRID_SIZE)]

    def get_debug_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the last processed frame with debug overlays
        (grid lines + detected labels) drawn on it, or None if no frame
        has been processed yet."""
        if self.last_frame is None or self.last_matrix is None:
            return None
        fc = self.last_frame.copy()
        draw_debug(fc, self.last_matrix)
        return fc

    def _run(self) -> None:
        """Dispatch to the appropriate run loop based on configured mode."""
        if self.demo:
            self._run_demo()
        elif self._frame_provider is not None:
            self._run_camera_shared()
        else:
            self._run_camera_own()

    def _run_demo(self) -> None:
        """Run loop for demo mode: repeatedly renders the synthetic frame,
        classifies it, and stabilizes the result."""
        while self._running:
            frame            = make_screen_simulation_frame()
            raw, _           = detect_matrix(frame)
            stable           = self._stable_matrix.update(raw)
            self.last_frame  = frame.copy()
            self.last_matrix = stable
            time.sleep(0.5)

    def _run_camera_shared(self) -> None:
        """Run loop pulling frames from an externally supplied frame
        provider, classifying, stabilizing, and firing on_matrix_ready
        once the result is stable across STABLE_WINDOW consecutive
        identical stable matrices."""
        prev_matrix    = None
        stable_counter = 0
        while self._running:
            try:
                frame = self._frame_provider()
            except Exception:
                time.sleep(0.05)
                continue
            if frame is None or frame.size == 0:
                time.sleep(0.05)
                continue
            roi              = _get_roi(frame)
            raw, _           = detect_matrix(roi)
            stable           = self._stable_matrix.update(raw)
            self.last_frame  = roi
            self.last_matrix = stable
            if stable == prev_matrix:
                stable_counter += 1
            else:
                stable_counter = 0
                prev_matrix    = stable
            all_valid = all(c != "empty" for row in stable for c in row)
            if all_valid and stable_counter >= STABLE_WINDOW and not self._matrix_sent:
                self._matrix_sent = True
                self.on_matrix_ready(stable)
            time.sleep(0.033)

    def _run_camera_own(self) -> None:
        """Run loop reading directly from an owned cv2.VideoCapture
        device. Automatically attempts to reopen the camera after
        _MAX_READ_FAILURES consecutive failed reads."""
        cap = _open_capture(self.camera_index)
        if cap is None:
            cap = cv2.VideoCapture(self.camera_index)

        prev_matrix          = None
        stable_counter       = 0
        consecutive_failures = 0

        while self._running:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                consecutive_failures += 1
                if consecutive_failures >= _MAX_READ_FAILURES:
                    cap.release()
                    time.sleep(0.5)
                    cap = _open_capture(self.camera_index) or cv2.VideoCapture(self.camera_index)
                    consecutive_failures = 0
                time.sleep(0.05)
                continue
            consecutive_failures = 0

            roi    = _get_roi(frame)
            raw, _ = detect_matrix(roi)
            stable = self._stable_matrix.update(raw)
            self.last_frame  = roi
            self.last_matrix = stable

            if stable == prev_matrix:
                stable_counter += 1
            else:
                stable_counter = 0
                prev_matrix    = stable

            all_valid = all(c != "empty" for row in stable for c in row)
            if all_valid and stable_counter >= STABLE_WINDOW and not self._matrix_sent:
                self._matrix_sent = True
                self.on_matrix_ready(stable)
            time.sleep(0.033)
        cap.release()
