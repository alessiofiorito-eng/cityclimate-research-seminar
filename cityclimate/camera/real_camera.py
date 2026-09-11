# camera/real_camera.py
"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: camera/real_camera.py — RealCamera, a live webcam source
(via cv2.VideoCapture) with a single background frame-grab thread.
Provides thread-safe cached-frame access, homography-based grid
extraction, manual cell/grid overrides, device scanning/switching, and
graceful fallback (placeholder frame/empty grid) when no camera is
available.
"""
from __future__ import annotations

import copy
import json
import logging
import platform
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

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
_OPEN_TIMEOUT_MS     = 1500   # reduced: faster scan
_MAX_READ_FAILURES   = 5


_PREFERRED_BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
_FALLBACK_BACKEND  = cv2.CAP_MSMF  if platform.system() == "Windows" else cv2.CAP_ANY


def _make_fallback_frame() -> np.ndarray:
    """Render a plain dark-gray placeholder frame with a "No Camera"
    label, used whenever no real frame is available."""
    frame = np.full((_FALLBACK_FRAME_SIZE, _FALLBACK_FRAME_SIZE, 3), 40, dtype=np.uint8)
    cv2.putText(
        frame, "No Camera",
        (int(_FALLBACK_FRAME_SIZE * 0.28), int(_FALLBACK_FRAME_SIZE * 0.52)),
        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (180, 180, 180), 3, cv2.LINE_AA,
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
        logger.warning("RealCamera: calibration file '%s' not found.", path)
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        raw = data.get("homography_matrix") or data.get("homography")
        if raw is None:
            logger.error("RealCamera: calibration.json missing 'homography_matrix'. Keys: %s", list(data.keys()))
            return None
        matrix = np.array(raw, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"Expected 3x3, got {matrix.shape}")
        logger.info("RealCamera: loaded homography from '%s'.", path)
        return matrix
    except Exception as exc:
        logger.error("RealCamera: failed to load homography — %s", exc)
        return None


def _try_open(index: int, backend: int) -> Optional[cv2.VideoCapture]:
    """Try to open a VideoCapture; return None on failure."""
    try:
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _OPEN_TIMEOUT_MS)
        if cap.open(index, backend) and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        cap.release()
    except Exception:
        pass
    return None


def _open_capture(index: int) -> Optional[cv2.VideoCapture]:
    """Try preferred backend, then fallback."""
    for backend in (_PREFERRED_BACKEND, _FALLBACK_BACKEND):
        cap = _try_open(index, backend)
        if cap is not None:
            bname = {cv2.CAP_DSHOW: "CAP_DSHOW", cv2.CAP_MSMF: "CAP_MSMF"}.get(backend, "CAP_ANY")
            logger.info("RealCamera: opened device %d via %s.", index, bname)
            return cap
    logger.warning("RealCamera: could not open device %d.", index)
    return None


def scan_cameras(max_index: int = 6, timeout_ms: int = _OPEN_TIMEOUT_MS) -> list[int]:
    """Scan for available cameras. Runs in a background thread (QThread) in admin_panel."""
    available: list[int] = []
    for idx in range(max_index):
        try:
            cap = cv2.VideoCapture()
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            if cap.open(idx, _PREFERRED_BACKEND) and cap.isOpened():
                available.append(idx)
            cap.release()
        except Exception:
            pass
    return available


class RealCamera:
    """Live webcam source with a single background grab thread.

    Only the internal ``_grab_loop`` thread ever calls ``cap.read()``.
    All callers (GUI timer, LiveStreamDetector, AdminPanel) use
    ``get_frame()`` which returns a copy of the last cached frame —
    no concurrent cap.read(), no races, no crashes.
    """

    def __init__(
        self,
        device_index: int = 0,
        calibration_file: str = CALIBRATION_FILE,
    ) -> None:
        """Load calibration (if available), initialize fallback state,
        and start the background grab thread for the given device.

        Args:
            device_index: cv2.VideoCapture device index to open.
            calibration_file: Path to the homography calibration file.
        """
        self._lock             = threading.Lock()
        self._device_index     = device_index
        self._calibration_file = calibration_file

        self._manual_grid: Optional[list[list[str | None]]] = None
        self._last_grid: list[list[str | None]]  = _empty_grid()
        self._last_frame: np.ndarray             = _make_fallback_frame()
        self._homography: Optional[np.ndarray]   = _load_homography(calibration_file)

        # Cap is owned by the grab thread; we keep a reference only for
        # is_open() queries — never call cap.read() from outside _grab_loop.
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._grab_thread: Optional[threading.Thread] = None

        self._start(device_index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self) -> np.ndarray:
        """Return latest cached BGR frame (never blocks on cap.read)."""
        with self._lock:
            return self._last_frame.copy()

    def get_grid(self) -> list[list[str | None]]:
        """Return the current logical LCZ grid.

        If a manual override grid is pending (set via `set_grid` or
        `update_cell`), it is consumed and returned once; otherwise the
        grid is (re-)extracted from the current camera frame via
        homography warp + color classification.

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
            logger.info("RealCamera: homography reloaded.")
            return True
        logger.warning("RealCamera: reload_calibration() — no valid homography.")
        return False

    def switch_device(self, device_index: int) -> bool:
        """Stop grab thread, release old cap, open new device."""
        logger.info("RealCamera: switching to device %d.", device_index)
        self._stop()          # stops thread AND releases cap safely
        with self._lock:
            self._last_frame = _make_fallback_frame()
        ok = self._start(device_index)
        logger.info("RealCamera: switch_device(%d) -> %s.", device_index, "OK" if ok else "FAILED")
        return ok

    def is_open(self) -> bool:
        """Return True if the underlying VideoCapture is currently open."""
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Stop the grab thread and release the camera device."""
        self._stop()
        logger.info("RealCamera: released.")

    # ------------------------------------------------------------------
    # Internal — lifecycle
    # ------------------------------------------------------------------

    def _start(self, index: int) -> bool:
        """Open the given device index and, if successful, start the
        background grab thread reading from it.

        Args:
            index: cv2.VideoCapture device index to open.

        Returns:
            True if the device was opened and the grab thread started,
            False if the device could not be opened.
        """
        cap = _open_capture(index)
        # Store cap reference (for is_open) under lock
        with self._lock:
            self._cap          = cap
            self._device_index = index
        if cap is None:
            return False
        self._running      = True
        self._grab_thread  = threading.Thread(target=self._grab_loop, args=(cap,), daemon=True)
        self._grab_thread.start()
        return True

    def _stop(self) -> None:
        """Signal grab thread to stop and wait for it, then release cap.

        Cap is released AFTER the thread exits — guarantees no concurrent
        cap.read() when we call cap.release(), which prevents the
        cv2.error: Unknown C++ exception on Windows.
        """
        self._running = False

        # Grab thread ref and cap ref atomically, clear them
        with self._lock:
            thread = self._grab_thread
            cap    = self._cap
            self._grab_thread = None
            self._cap         = None

        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)

        # Release cap only after thread has exited
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:
                logger.warning("RealCamera: cap.release() raised — %s", exc)

    # ------------------------------------------------------------------
    # Grab loop (runs in background thread)
    # ------------------------------------------------------------------

    def _grab_loop(self, cap: cv2.VideoCapture) -> None:
        """Sole caller of cap.read(). cap ownership is passed in; we do NOT
        touch self._cap here to avoid any lock conflicts with _stop()."""
        failures = 0
        while self._running:
            try:
                ret, frame = cap.read()
            except Exception as exc:
                logger.warning("RealCamera: cap.read() raised — %s", exc)
                time.sleep(0.1)
                continue

            if not ret or frame is None or frame.size == 0:
                failures += 1
                if failures >= _MAX_READ_FAILURES:
                    logger.warning(
                        "RealCamera: %d consecutive read failures on device %d — stopping.",
                        failures, self._device_index,
                    )
                    # Don't try to re-open here; let switch_device handle recovery.
                    break
                time.sleep(0.05)
                continue

            failures = 0
            with self._lock:
                self._last_frame = frame
            time.sleep(0.01)   # ~100 fps cap to avoid busy-loop

        logger.info("RealCamera: grab thread exited (device %d).", self._device_index)
        # Do NOT release cap here — _stop() does it after join()

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
            logger.error("RealCamera: extract_grid() raised — %s", exc)
            return None
