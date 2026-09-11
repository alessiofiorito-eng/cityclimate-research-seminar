"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: camera/realsense_camera.py — RealSenseCamera, an Intel
RealSense RGB stream wrapper using pyrealsense2. Drop-in replacement
for RealCamera when cv2.VideoCapture cannot access the RealSense color
stream (common on Windows).

Usage:
    python main.py --realsense
"""
from __future__ import annotations

import copy
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

from config import (
    GRID_ROWS,
    GRID_COLS,
    BOARD_WIDTH_PX,
    BOARD_HEIGHT_PX,
    CALIBRATION_FILE,
)
from vision.grid_extractor import extract_grid

logger = logging.getLogger(__name__)


_FALLBACK_FRAME_SIZE = 800


def _make_fallback_frame() -> np.ndarray:
    """Render a plain dark-gray placeholder frame with a "No RealSense"
    label, used whenever no live RealSense stream is available."""
    import cv2
    frame = np.full((_FALLBACK_FRAME_SIZE, _FALLBACK_FRAME_SIZE, 3), 40, dtype=np.uint8)
    cv2.putText(
        frame, "No RealSense",
        (int(_FALLBACK_FRAME_SIZE * 0.18), int(_FALLBACK_FRAME_SIZE * 0.52)),
        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (180, 180, 180), 3, cv2.LINE_AA,
    )
    return frame


def _empty_grid() -> list[list[str | None]]:
    """Return a GRID_ROWS x GRID_COLS grid of all None (empty cells)."""
    return [[None] * GRID_COLS for _ in range(GRID_ROWS)]


def _load_homography(path: str = CALIBRATION_FILE) -> Optional[np.ndarray]:
    """Load and validate the saved 3x3 homography matrix from the
    calibration JSON file.

    Args:
        path: Path to the calibration file.

    Returns:
        The homography as a 3x3 float64 numpy array, or None if the
        file is missing, malformed, or contains an invalid matrix
        shape.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("RealSenseCamera: calibration file '%s' not found.", path)
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        raw  = data.get("homography_matrix") or data.get("homography")
        if raw is None:
            logger.error("RealSenseCamera: calibration.json missing 'homography_matrix'.")
            return None
        matrix = np.array(raw, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"Expected 3x3, got {matrix.shape}")
        logger.info("RealSenseCamera: loaded homography from '%s'.", path)
        return matrix
    except Exception as exc:
        logger.error("RealSenseCamera: failed to load homography — %s", exc)
        return None


class RealSenseCamera:
    """
    Thread-safe RealSense RGB stream via pyrealsense2.
    Mirrors the public API of RealCamera so it is a drop-in replacement.
    """

    def __init__(
        self,
        width:  int = 640,
        height: int = 480,
        fps:    int = 30,
        calibration_file: str = CALIBRATION_FILE,
    ) -> None:
        """Load calibration (if available) and start the RealSense RGB
        stream and its background read thread.

        Args:
            width: Requested color stream width in pixels.
            height: Requested color stream height in pixels.
            fps: Requested color stream frame rate.
            calibration_file: Path to the homography calibration file.

        Raises:
            RuntimeError: If the pyrealsense2 package is not installed.
        """
        if not _RS_AVAILABLE:
            raise RuntimeError(
                "pyrealsense2 ist nicht installiert.\n"
                "Bitte ausfuehren:  pip install pyrealsense2"
            )
        self._width  = width
        self._height = height
        self._fps    = fps
        self._calibration_file = calibration_file

        self._lock    = threading.Lock()
        self._latest: np.ndarray             = _make_fallback_frame()
        self._homography: Optional[np.ndarray] = _load_homography(calibration_file)
        self._last_grid: list[list[str | None]] = _empty_grid()
        self._manual_grid: Optional[list[list[str | None]]] = None

        self._pipeline: Optional[rs.pipeline] = None
        self._running  = False
        self._thread:  Optional[threading.Thread] = None

        self._start()

    # ------------------------------------------------------------------
    # Public API  (mirrors RealCamera)
    # ------------------------------------------------------------------

    def get_frame(self) -> np.ndarray:
        """Return the latest cached BGR color frame (never blocks on the
        RealSense pipeline)."""
        with self._lock:
            return self._latest.copy()

    def get_grid(self) -> list[list[str | None]]:
        """Return the current logical LCZ grid.

        If a manual override grid is pending (set via `set_grid` or
        `update_cell`), it is consumed and returned once; otherwise the
        grid is (re-)extracted from the current frame via homography
        warp + color classification.

        Returns:
            A deep copy of the resulting GRID_ROWS x GRID_COLS grid.
        """
        with self._lock:
            if self._manual_grid is not None:
                grid = copy.deepcopy(self._manual_grid)
                self._manual_grid = None
                self._last_grid   = grid
                return copy.deepcopy(grid)
        grid = self._extract_via_cv()
        with self._lock:
            if grid is not None:
                self._last_grid = grid
            return copy.deepcopy(self._last_grid)

    def set_grid(self, grid: list[list[str | None]]) -> None:
        """Queue a full manual grid override, to be returned by the
        next `get_grid()` call instead of a CV-extracted grid."""
        with self._lock:
            self._manual_grid = copy.deepcopy(grid)

    def update_cell(self, row: int, col: int, lcz_id: str | None) -> None:
        """Apply a single-cell manual override on top of the last known
        grid (initializing the manual override grid from it if none is
        pending yet).

        Args:
            row: Row index of the cell to update.
            col: Column index of the cell to update.
            lcz_id: New LCZ ID for the cell, or None to clear it.

        Raises:
            IndexError: If (row, col) is outside the configured grid
                dimensions.
        """
        if not (0 <= row < GRID_ROWS and 0 <= col < GRID_COLS):
            raise IndexError(f"Cell ({row},{col}) out of bounds.")
        with self._lock:
            if self._manual_grid is None:
                self._manual_grid = copy.deepcopy(self._last_grid)
            self._manual_grid[row][col] = lcz_id

    def reload_calibration(self) -> bool:
        """Re-read the homography from the calibration file and replace
        the currently cached one.

        Returns:
            True if a valid homography was loaded, False otherwise.
        """
        h = _load_homography(self._calibration_file)
        with self._lock:
            self._homography = h
        if h is not None:
            logger.info("RealSenseCamera: homography reloaded.")
            return True
        logger.warning("RealSenseCamera: reload_calibration() — no valid homography.")
        return False

    def is_open(self) -> bool:
        """Return True if the RealSense pipeline is currently running."""
        return self._running and self._pipeline is not None

    def release(self) -> None:
        """Stop the read thread and the RealSense pipeline."""
        self._stop()
        logger.info("RealSenseCamera: released.")

    # ------------------------------------------------------------------
    # Internal — lifecycle
    # ------------------------------------------------------------------

    def _start(self) -> None:
        """Configure and start the RealSense color stream, enable auto
        white-balance/exposure, warm up the auto-exposure by discarding
        the first frames, then start the background read thread. Falls
        back to placeholder-frame mode (without raising) if the
        pipeline fails to start."""
        try:
            cfg = rs.config()
            cfg.enable_stream(
                rs.stream.color,
                self._width, self._height,
                rs.format.bgr8, self._fps,
            )
            pipeline = rs.pipeline()
            profile  = pipeline.start(cfg)

            # Enable auto white-balance and auto exposure for correct colors
            color_sensor = profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)

            # Warm up: let auto-exposure stabilise (skip first 30 frames)
            logger.info("RealSenseCamera: warming up auto-exposure...")
            for _ in range(30):
                pipeline.wait_for_frames(timeout_ms=2000)

            self._pipeline = pipeline
            self._running  = True
            self._thread   = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            logger.info(
                "RealSenseCamera: started RGB stream %dx%d @ %dfps.",
                self._width, self._height, self._fps,
            )
        except Exception as exc:
            logger.error("RealSenseCamera: failed to start — %s", exc)
            logger.warning("RealSenseCamera: running in fallback mode (no live feed).")

    def _stop(self) -> None:
        """Signal the read thread to stop, wait for it to exit, and stop
        the RealSense pipeline."""
        self._running = False
        thread = self._thread
        self._thread = None
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    def _loop(self) -> None:
        """Background thread loop: repeatedly wait for and fetch the
        latest color frame from the RealSense pipeline, caching it for
        `get_frame()` under the lock."""
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
                color  = frames.get_color_frame()
                if color:
                    img = np.asanyarray(color.get_data())
                    with self._lock:
                        self._latest = img
            except Exception as exc:
                logger.warning("RealSenseCamera: frame error — %s", exc)
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # CV extraction
    # ------------------------------------------------------------------

    def _extract_via_cv(self) -> Optional[list[list[str | None]]]:
        """Extract the LCZ grid from the current frame via homography
        warp + color classification (vision.grid_extractor.extract_grid),
        returning None if no calibration is available or extraction
        fails."""
        with self._lock:
            homography = self._homography
        if homography is None:
            return None
        frame = self.get_frame()
        try:
            return extract_grid(frame, homography, GRID_ROWS, GRID_COLS)
        except Exception as exc:
            logger.error("RealSenseCamera: extract_grid() raised — %s", exc)
            return None
