"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: vision/calibration.py — Homography persistence and computation.
Provides functions to compute a perspective homography from 4 manually
clicked corner points, and to save/load that homography to/from a JSON
calibration file so the board warp survives across application restarts.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import cv2
import numpy as np

from config import BOARD_WIDTH_PX, BOARD_HEIGHT_PX, CALIBRATION_FILE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default homography (identity mapping — used in demo mode without calibration)
# ---------------------------------------------------------------------------

DEFAULT_HOMOGRAPHY: np.ndarray = np.eye(3, dtype=np.float64)

# Canonical destination corners (top-left, top-right, bottom-right, bottom-left)
_DST_CORNERS: np.ndarray = np.array(
    [
        [0,                 0               ],
        [BOARD_WIDTH_PX - 1, 0              ],
        [BOARD_WIDTH_PX - 1, BOARD_HEIGHT_PX - 1],
        [0,                 BOARD_HEIGHT_PX - 1],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_homography(src_points: list[tuple[float, float]]) -> np.ndarray:
    """Compute a 3×3 homography from 4 user-clicked source points.

    Parameters
    ----------
    src_points:
        Exactly 4 (x, y) pixel coordinates in the raw camera frame,
        clicked in order: top-left, top-right, bottom-right, bottom-left.

    Returns
    -------
    3×3 float64 homography matrix (maps src → canonical board rectangle).

    Raises
    ------
    ValueError
        If fewer or more than 4 points are provided.
    """
    if len(src_points) != 4:
        raise ValueError(
            f"compute_homography requires exactly 4 source points, got {len(src_points)}."
        )

    src = np.array(src_points, dtype=np.float32)
    H, mask = cv2.findHomography(src, _DST_CORNERS, method=0)  # least-squares

    if H is None:
        logger.error("cv2.findHomography returned None — returning identity.")
        return DEFAULT_HOMOGRAPHY.copy()

    logger.info("Homography computed from %d points.", len(src_points))
    return H.astype(np.float64)


def save_calibration(
    matrix: np.ndarray,
    path: str = CALIBRATION_FILE,
) -> None:
    """Persist a homography matrix to a JSON file.

    Parameters
    ----------
    matrix:
        3×3 numpy array.
    path:
        Destination file path (default: CALIBRATION_FILE from config).
    """
    data = {"homography": matrix.tolist()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Calibration saved to '%s'.", path)


def load_calibration(path: str = CALIBRATION_FILE) -> np.ndarray | None:
    """Load a homography matrix from a JSON file.

    Parameters
    ----------
    path:
        Source file path (default: CALIBRATION_FILE from config).

    Returns
    -------
    3×3 float64 numpy array, or None if the file does not exist or is invalid.
    """
    if not os.path.isfile(path):
        logger.warning("Calibration file '%s' not found — using default homography.", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        matrix = np.array(data["homography"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"Expected (3,3) matrix, got {matrix.shape}.")
        logger.info("Calibration loaded from '%s'.", path)
        return matrix
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to load calibration from '%s': %s", path, exc)
        return None
