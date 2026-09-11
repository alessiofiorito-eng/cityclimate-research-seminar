"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: vision/grid_extractor.py — LEGACY STUB, do not use for live
detection. Grid extraction (warp + cell slicing) is now performed
internally by ``vision.cv_live_stream.LiveStreamDetector``. This file is
kept only so existing imports do not break during transition.

``extract_grid`` is provided here for RealCamera._extract_via_cv(): it
warps the raw frame with the calibration homography, resizes to the
pipeline's canonical FRAME_SIZE, then delegates to detect_matrix.
"""
# ruff: noqa
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from vision.cv_live_stream import (
    FRAME_SIZE,
    detect_matrix,
    make_screen_simulation_frame,
    COLOR_TO_LCZ,
)


def extract_grid(
    frame: np.ndarray,
    homography: np.ndarray,
    rows: int,
    cols: int,
) -> list[list[str | None]]:
    """Warp *frame* with *homography*, classify all cells, return LCZ grid.

    Args:
        frame:       Raw BGR frame from the camera.
        homography:  3x3 perspective transform matrix (from calibration.json).
        rows:        Number of grid rows (GRID_ROWS from config).
        cols:        Number of grid columns (GRID_COLS from config).

    Returns:
        GRID_ROWS x GRID_COLS list of LCZ-ID strings or None for empty cells.
    """
    # 1. Warp the raw frame to the canonical board view
    warped = cv2.warpPerspective(
        frame, homography, (FRAME_SIZE, FRAME_SIZE),
        flags=cv2.INTER_LINEAR,
    )

    # 2. Classify all cells using the shared HSV detection pipeline
    color_matrix, _ = detect_matrix(warped)

    # 3. Convert color names -> LCZ IDs (None for "empty" cells)
    lcz_grid: list[list[str | None]] = [
        [COLOR_TO_LCZ.get(cell) for cell in row]
        for row in color_matrix
    ]
    return lcz_grid


def diff_grids(
    old: list[list[Optional[str]]],
    new: list[list[Optional[str]]],
) -> list[dict]:
    """Return a list of changed cells between two grids.

    Compares two same-shaped LCZ grids cell by cell and collects every
    position whose value differs.

    Args:
        old: Previous LCZ grid (rows of LCZ-ID strings or None).
        new: Current LCZ grid to compare against `old`.

    Returns:
        A list of change records, each ``{"row": r, "col": c, "old":
        old_val, "new": new_val}``.
    """
    changes: list[dict] = []
    for r, (old_row, new_row) in enumerate(zip(old, new)):
        for c, (o, n) in enumerate(zip(old_row, new_row)):
            if o != n:
                changes.append({"row": r, "col": c, "old": o, "new": n})
    return changes


__all__ = ["extract_grid", "detect_matrix", "make_screen_simulation_frame", "diff_grids"]
