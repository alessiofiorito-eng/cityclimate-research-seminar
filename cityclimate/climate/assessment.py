"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: climate/assessment.py — Climate metrics computation for the
CityClimate Board. Applies neighbour cooling effects to a raw LCZ grid,
then derives the Heat Exposure (HE) index, mean temperature, total
population, and green/built land-cover fractions for the current board
configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from config import LCZ_CLASSES, GRID_ROWS, GRID_COLS, CELL_AREA_KM2, LCZ_BASELINE_ID

logger = logging.getLogger(__name__)

# LCZ IDs counted as "green" land cover (vegetation/water) for green_fraction
GREEN_LCZ_IDS:  frozenset[str] = frozenset({"A", "D", "G"})
# LCZ IDs counted as "built" land cover (urban fabric) for built_fraction
BUILT_LCZ_IDS:  frozenset[str] = frozenset({"2", "5", "9", "10"})

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ClimateMetrics:
    """Aggregated climate metrics for a single grid configuration.

    Attributes:
        heat_exposure: Population-weighted Heat Exposure index (°C),
            relative to the LCZ baseline class.
        mean_temp: Unweighted mean temperature (°C) across all grid cells.
        total_population: Estimated total population represented by the
            grid (persons), derived from per-cell population density.
        green_fraction: Fraction of grid cells classified as green land
            cover (vegetation/water), in [0, 1].
        built_fraction: Fraction of grid cells classified as built-up
            urban land cover, in [0, 1].
        temp_grid: 2-D array (as nested lists) of per-cell temperatures
            after cooling effects have been applied. Excluded from the
            dataclass repr for readability.
        pop_grid: 2-D array (as nested lists) of per-cell population
            density values. Excluded from the dataclass repr for
            readability.
    """
    heat_exposure:    float
    mean_temp:        float
    total_population: float
    green_fraction:   float
    built_fraction:   float
    temp_grid:        list[list[float]] = field(repr=False)
    pop_grid:         list[list[float]] = field(repr=False)


# ---------------------------------------------------------------------------
# Step 1 — cooling propagation (4-connected Von-Neumann neighbours only,
#          maximum of all neighbour sources — no additive stacking)
# ---------------------------------------------------------------------------

def apply_cooling(grid: list[list[str | None]]) -> list[list[float]]:
    """Return a 2-D temperature grid after applying neighbour cooling effects.

    Rules (revised):
      1. Fill base temperatures from LCZ_CLASSES[id].rel_temp.
      2. For every cell with cooling != 0, propagate abs(cooling) to the 4
         orthogonal (Von-Neumann) neighbours only — NOT diagonals.
      3. Each neighbour takes the MAXIMUM cooling it receives from any single
         source — effects do NOT stack additively.

    Args:
        grid: 2-D list of LCZ IDs (str) or None for empty cells, laid out
            as grid[row][col].

    Returns:
        A 2-D list (same shape as grid) of resulting temperatures in °C,
        relative to the configured LCZ baseline class, after cooling has
        been subtracted.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    # Start every cell at the baseline LCZ's temperature, then overwrite
    # cells that actually have an assigned LCZ class below.
    baseline_temp: float = LCZ_CLASSES[LCZ_BASELINE_ID]["rel_temp"]
    temp = np.full((rows, cols), baseline_temp, dtype=float)

    for r in range(rows):
        for c in range(cols):
            lcz_id = grid[r][c]
            if lcz_id is not None and lcz_id in LCZ_CLASSES:
                temp[r, c] = LCZ_CLASSES[lcz_id]["rel_temp"]

    # Track the maximum cooling contribution each cell receives
    max_cooling = np.zeros((rows, cols), dtype=float)

    # 4-connected (Von-Neumann) offsets only
    VON_NEUMANN = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(rows):
        for c in range(cols):
            lcz_id = grid[r][c]
            if lcz_id is None or lcz_id not in LCZ_CLASSES:
                continue
            effect = abs(LCZ_CLASSES[lcz_id]["cooling"])   # positive magnitude
            if effect == 0.0:
                continue
            # Propagate this source's cooling magnitude to each orthogonal
            # neighbour, keeping only the strongest effect per neighbour
            # (no additive stacking of multiple cooling sources).
            for dr, dc in VON_NEUMANN:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if effect > max_cooling[nr, nc]:
                        max_cooling[nr, nc] = effect

    temp -= max_cooling

    logger.debug(
        "Temperature grid computed (4-conn max-cooling). min=%.2f max=%.2f mean=%.2f",
        temp.min(), temp.max(), temp.mean(),
    )
    return temp.tolist()


# ---------------------------------------------------------------------------
# Step 2 — full metric computation
# ---------------------------------------------------------------------------

def compute_metrics(grid: list[list[str | None]]) -> ClimateMetrics:
    """Compute all ClimateMetrics for a given LCZ grid.

    Heat Exposure follows the paper formula exactly:
        HE = Σ(Pᵢ · Tᵢ) / Σ(Pᵢ)
    where Pᵢ = pop_density of cell i, Tᵢ = cooled temperature of cell i.
    If Σ(Pᵢ) == 0 (no inhabited cells), falls back to mean temperature.

    Args:
        grid: 2-D list of LCZ IDs (str) or None for empty cells, laid out
            as grid[row][col].

    Returns:
        A populated ClimateMetrics instance describing the current grid
        configuration.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    total_cells = rows * cols

    temp_grid_2d = apply_cooling(grid)
    temp_arr = np.array(temp_grid_2d, dtype=float)

    # Build the per-cell population density array from the LCZ table
    pop_arr = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            lcz_id = grid[r][c]
            if lcz_id is not None and lcz_id in LCZ_CLASSES:
                pop_arr[r, c] = LCZ_CLASSES[lcz_id]["pop_density"]

    # HE = Σ(Pᵢ·Tᵢ) / Σ(Pᵢ)  — only inhabited cells contribute
    sum_pop = float(pop_arr.sum())
    if sum_pop > 0.0:
        heat_exposure = float((pop_arr * temp_arr).sum() / sum_pop)
    else:
        # No population anywhere on the board: fall back to the plain
        # (unweighted) mean temperature instead of dividing by zero.
        heat_exposure = float(temp_arr.mean())
        logger.debug("No population; HE falls back to mean temperature.")

    total_population = sum_pop * CELL_AREA_KM2
    mean_temp = float(temp_arr.mean())

    # Count cells belonging to the green / built LCZ groups to derive
    # land-cover fractions.
    green_count = sum(
        1 for r in range(rows) for c in range(cols) if grid[r][c] in GREEN_LCZ_IDS
    )
    built_count = sum(
        1 for r in range(rows) for c in range(cols) if grid[r][c] in BUILT_LCZ_IDS
    )

    green_fraction = green_count / total_cells if total_cells else 0.0
    built_fraction = built_count / total_cells if total_cells else 0.0

    metrics = ClimateMetrics(
        heat_exposure=heat_exposure,
        mean_temp=mean_temp,
        total_population=total_population,
        green_fraction=green_fraction,
        built_fraction=built_fraction,
        temp_grid=temp_grid_2d,
        pop_grid=pop_arr.tolist(),
    )

    logger.info(
        "Metrics — HE: %.2f°C | mean_T: %.2f°C | pop: %.0f | green: %.1f%% | built: %.1f%%",
        metrics.heat_exposure, metrics.mean_temp, metrics.total_population,
        metrics.green_fraction * 100, metrics.built_fraction * 100,
    )
    return metrics


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick manual sanity check: compute metrics for the "urban_heat_island"
    # sample grid and print the key results to the console.
    import sys, os
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tests.sample_grids import SAMPLE_GRIDS
    grid = SAMPLE_GRIDS["urban_heat_island"]
    m = compute_metrics(grid)
    print(f"  HE: {m.heat_exposure:+.4f} °C")
    print(f"  Mean T: {m.mean_temp:+.4f} °C")
    print(f"  Pop: {m.total_population:,.0f}")
