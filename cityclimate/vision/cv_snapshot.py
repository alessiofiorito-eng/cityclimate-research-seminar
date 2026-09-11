"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: vision/cv_snapshot.py — Single-shot (as opposed to continuous)
board detection. Provides SnapshotDetector, which captures one frame
(from a real camera or the synthetic screen-simulation) on demand and
runs it through the shared color-detection pipeline from
vision.cv_live_stream.
"""

import cv2
import numpy as np
import platform
from typing import Optional, List

from vision.cv_live_stream import (
    FRAME_SIZE, GRID_SIZE, CELL_SIZE,
    detect_matrix, draw_debug, make_screen_simulation_frame, COLOR_RANGES,
    _open_capture,
)


class SnapshotDetector:
    """Single-shot detector. Call take_snapshot() to capture the current
    board state. Works with real camera or screen simulation (demo=True).
    """

    def __init__(self, demo: bool = False, camera_index: int = 0):
        """Configure the detector.

        Args:
            demo: If True, use the synthetic screen-simulation frame
                instead of a real camera.
            camera_index: cv2.VideoCapture device index to use when not
                in demo mode.
        """
        self.demo = demo
        self.camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self):
        """Open the underlying camera device (no-op in demo mode).

        Tries the shared `_open_capture` helper first (which prefers
        platform-specific backends), falling back to a plain
        cv2.VideoCapture if that fails.
        """
        if not self.demo:
            self._cap = _open_capture(self.camera_index)
            if self._cap is None:
                # Hard fallback
                self._cap = cv2.VideoCapture(self.camera_index)

    def close(self):
        """Release the underlying camera device, if open."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def take_snapshot(self, sim_matrix: Optional[List[List[str]]] = None):
        """Capture and classify a single frame of the board.

        In demo mode, renders a synthetic frame (optionally overridden by
        sim_matrix); otherwise reads one frame from the camera, center-
        crops it to FRAME_SIZE, and guards against frames that are too
        small for the grid.

        Args:
            sim_matrix: Optional color-name matrix to render instead of
                the default synthetic demo board (demo mode only).

        Returns:
            A tuple (matrix, debug_frame) where matrix is the detected
            GRID_SIZE x GRID_SIZE color matrix and debug_frame is the
            captured frame annotated with grid lines and labels. Returns
            (None, None) if no usable frame could be captured.
        """
        if self.demo:
            frame = make_screen_simulation_frame(sim_matrix)
        else:
            if self._cap is None or not self._cap.isOpened():
                self.open()
            ret, frame = self._cap.read()
            if not ret or frame is None or frame.size == 0:
                return None, None
            h, w = frame.shape[:2]
            xf = max(0, (w - FRAME_SIZE) // 2)
            yf = max(0, (h - FRAME_SIZE) // 2)
            frame = frame[yf:yf + FRAME_SIZE, xf:xf + FRAME_SIZE].copy()
            # Guard: camera resolution too low for the grid
            if frame.shape[0] < FRAME_SIZE or frame.shape[1] < FRAME_SIZE:
                return None, None

        matrix, valid = detect_matrix(frame)
        debug_frame = frame.copy()
        draw_debug(debug_frame, matrix)
        return matrix, debug_frame
