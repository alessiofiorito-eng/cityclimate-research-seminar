# camera/mock_camera.py — Software mock camera for CityClimate Board
"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: camera/mock_camera.py — MockCamera, a software-only stand-in
for a physical camera. Renders the current logical LCZ grid directly
as a synthetic BGR frame (with grid lines and optional pixel noise),
letting the rest of the application run in demo mode without any real
camera hardware or board.
"""

from __future__ import annotations

import copy
import logging

import cv2
import numpy as np

from config import (
    GRID_ROWS,
    GRID_COLS,
    LCZ_CLASSES,
    MOCK_FRAME_SIZE,
    MOCK_NOISE_SIGMA,
    MOCK_DEFAULT_GRID,
)
from tests.sample_grids import DEMO_GRIDS

logger = logging.getLogger(__name__)


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string (e.g. "#8B0000") to an OpenCV-style
    BGR tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


_LCZ_BGR: dict[str, tuple[int, int, int]] = {
    lcz_id: _hex_to_bgr(info["color_hex"])
    for lcz_id, info in LCZ_CLASSES.items()
}


_UNKNOWN_BGR: tuple[int, int, int] = (80, 80, 80)


class MockCamera:
    """Renders an LCZ grid as a synthetic camera frame (no display window)."""

    def __init__(self, grid_name: str = MOCK_DEFAULT_GRID) -> None:
        """Initialize the mock camera with a starting demo grid.

        Args:
            grid_name: Key into DEMO_GRIDS selecting the initial grid
                to render.

        Raises:
            ValueError: If `grid_name` is not a known key in
                DEMO_GRIDS.
        """
        if grid_name not in DEMO_GRIDS:
            raise ValueError(
                f"Unknown grid '{grid_name}'. Available: {list(DEMO_GRIDS.keys())}"
            )
        self._grid: list[list[str | None]] = copy.deepcopy(DEMO_GRIDS[grid_name])
        self._frame: np.ndarray | None = None
        logger.info("MockCamera initialised with grid '%s'.", grid_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self) -> np.ndarray:
        """Return a cached BGR frame (MOCK_FRAME_SIZE x MOCK_FRAME_SIZE px)."""
        if self._frame is None:
            self._render()
        return self._frame.copy()

    def get_grid(self) -> list[list[str | None]]:
        """Return a deep-copy of the current logical grid.

        Use this in demo mode instead of extract_grid() to avoid
        running the colour classifier on the synthetic frame.
        """
        return copy.deepcopy(self._grid)

    def set_grid(self, grid: list[list[str | None]]) -> None:
        """Replace the entire grid and invalidate the frame cache."""
        self._grid = copy.deepcopy(grid)
        self._frame = None
        logger.debug("MockCamera: full grid replaced, cache invalidated.")

    def update_cell(self, row: int, col: int, lcz_id: str | None) -> None:
        """Update a single cell and invalidate the frame cache.

        Args:
            row: Row index of the cell to update.
            col: Column index of the cell to update.
            lcz_id: New LCZ ID for the cell, or None to clear it.

        Raises:
            IndexError: If (row, col) is outside the configured grid
                dimensions.
        """
        if not (0 <= row < GRID_ROWS and 0 <= col < GRID_COLS):
            raise IndexError(
                f"Cell ({row}, {col}) out of bounds for {GRID_ROWS}x{GRID_COLS} grid."
            )
        self._grid[row][col] = lcz_id
        self._frame = None
        logger.debug(
            "MockCamera: cell (%d, %d) set to '%s', cache invalidated.", row, col, lcz_id
        )

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Render the current logical grid into a synthetic BGR frame:
        fill each cell with its LCZ color, draw white grid lines between
        cells, and optionally add Gaussian pixel noise to more closely
        mimic a real camera capture."""
        size = MOCK_FRAME_SIZE
        frame = np.zeros((size, size, 3), dtype=np.uint8)

        cell_w = size / GRID_COLS
        cell_h = size / GRID_ROWS

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x0 = int(round(c * cell_w))
                y0 = int(round(r * cell_h))
                x1 = int(round((c + 1) * cell_w))
                y1 = int(round((r + 1) * cell_h))

                lcz_id = self._grid[r][c]
                bgr = _LCZ_BGR.get(lcz_id, _UNKNOWN_BGR)
                cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), bgr, thickness=-1)

        for c in range(1, GRID_COLS):
            x = int(round(c * cell_w))
            cv2.line(frame, (x, 0), (x, size - 1), (255, 255, 255), 1)
        for r in range(1, GRID_ROWS):
            y = int(round(r * cell_h))
            cv2.line(frame, (0, y), (size - 1, y), (255, 255, 255), 1)

        if MOCK_NOISE_SIGMA > 0:
            noise = np.random.normal(0, MOCK_NOISE_SIGMA, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        self._frame = frame
        logger.debug("MockCamera: frame rendered (%dx%d px).", size, size)
