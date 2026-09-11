# cityclimate/llm/matrix_context.py
"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: llm/matrix_context.py — MatrixContext, a snapshot dataclass
bundling the current board grid, its computed climate metrics, a
timestamp, and the data source (live camera, snapshot, or demo),
passed to the prompt-building pipeline in prompt_builder.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from climate.assessment import ClimateMetrics


@dataclass
class MatrixContext:
    """Snapshot of the current board state passed to the LLM pipeline."""

    grid: List[List[Optional[str]]]
    """5x5 grid of LCZ-IDs (e.g. '2', 'A', 'G') or None for empty cells."""

    metrics: ClimateMetrics
    """Computed climate metrics for this grid configuration."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp of when the snapshot was taken."""

    source: Literal["live", "snapshot", "demo"] = "demo"
    """Origin of the grid data: live camera, frozen snapshot, or demo."""

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                  #
    # ------------------------------------------------------------------ #

    @property
    def rows(self) -> int:
        """Number of grid rows."""
        return len(self.grid)

    @property
    def cols(self) -> int:
        """Number of grid columns (0 if the grid has no rows)."""
        return len(self.grid[0]) if self.grid else 0

    def lcz_counts(self) -> dict[str, int]:
        """Return a frequency dict of LCZ-IDs present in the grid."""
        counts: dict[str, int] = {}
        for row in self.grid:
            for cell in row:
                if cell is not None:
                    counts[cell] = counts.get(cell, 0) + 1
        return counts

    def empty_count(self) -> int:
        """Number of cells with no block placed."""
        return sum(1 for row in self.grid for cell in row if cell is None)
