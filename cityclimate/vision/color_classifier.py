"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: vision/color_classifier.py — LEGACY STUB, do not use for live
detection. All color classification is now handled by
``vision.cv_live_stream``. This file is kept only so existing imports do
not break during transition, and simply re-exports the relevant symbols
from ``vision.cv_live_stream``.
"""
# ruff: noqa
from __future__ import annotations

from vision.cv_live_stream import (
    COLOR_RANGES,
    COLOR_TO_LCZ,
    detect_cell_color,
    color_matrix_to_lcz,
)

__all__ = ["COLOR_RANGES", "COLOR_TO_LCZ", "detect_cell_color", "color_matrix_to_lcz"]
