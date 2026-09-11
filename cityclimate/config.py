"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: config.py — Global configuration for the CityClimate Board system.
Defines the physical grid layout, board display size, calibration file
location, mock camera parameters, Ollama LLM connection settings, the
color-to-LCZ mapping used by the computer vision pipeline, the Local
Climate Zone (LCZ) parameter table (temperature, population density,
cooling effect, height, imperviousness, sky view factor, and derived
HSV color values), the reachable KPI ranges used for UI scaling, and
the GUI color theme.
"""

import colorsys

# ---------------------------------------------------------------------------
# Grid — physical board is 5x5, each cell = 0.5 km x 0.5 km.
# See paper for details on the board dimensioning.
# ---------------------------------------------------------------------------
GRID_COLS: int = 5
GRID_ROWS: int = 5
CELL_AREA_KM2: float = 0.25  # Area per grid cell in km^2 (0.5 km x 0.5 km)

# ---------------------------------------------------------------------------
# Board display — canvas size (in pixels) the calibrated board is warped to.
# ---------------------------------------------------------------------------
BOARD_WIDTH_PX: int = 800
BOARD_HEIGHT_PX: int = 800

# ---------------------------------------------------------------------------
# Camera / calibration — path to the persisted homography/calibration data.
# ---------------------------------------------------------------------------
CALIBRATION_FILE: str = "calibration.json"

# ---------------------------------------------------------------------------
# Mock camera / screen simulation settings used when no physical camera or
# board is connected (development and demo mode).
# ---------------------------------------------------------------------------
MOCK_FRAME_SIZE: int = 480       # Size (px) of the synthetic square frame
MOCK_NOISE_SIGMA: float = 4.0    # Standard deviation of injected pixel noise
MOCK_DEFAULT_GRID: str = "green_city"  # Default demo grid preset to load
SCREEN_SIMULATION: bool = True   # Enable on-screen simulation mode

# ---------------------------------------------------------------------------
# Ollama LLM connection settings for the local language model backend.
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "climate-analyst"
OLLAMA_TIMEOUT: float = 60.0

# ---------------------------------------------------------------------------
# CV color -> LCZ mapping.
# SINGLE SOURCE OF TRUTH — cv_live_stream.py mirrors this exactly.
#
# Maps the detected tile color name (from the computer vision pipeline) to
# the corresponding Local Climate Zone (LCZ) class identifier used
# throughout the rest of the application. "empty" (no tile detected) maps
# to None.
#
# Physical board tile legend:
#   red         -> LCZ 2 (Compact mid-rise, dark red tile)
#   lightgray   -> LCZ 5 (Open mid-rise, light grey tile)
#   white       -> LCZ 6 (Open low-rise, white tile)
#   darkgray    -> LCZ 9 (Sparsely built, dark grey tile)
#   darkgreen   -> LCZ A (Dense trees, dark green tile)
#   lightgreen  -> LCZ D (Low plants, light green tile)
#   blue        -> LCZ G (Water, blue tile)
# ---------------------------------------------------------------------------
COLOR_TO_LCZ: dict[str, str | None] = {
    "red": "2",
    "lightgray": "5",
    "white": "6",
    "darkgray": "9",
    "darkgreen": "A",
    "lightgreen": "D",
    "blue": "G",
    "empty": None,
}

# ---------------------------------------------------------------------------
# Tile display colors — match the physical board tiles exactly.
# Used by GridView AND the LCZ distribution chart so that the on-screen
# colors are always consistent with the physical board.
# ---------------------------------------------------------------------------
LCZ_TILE_COLORS: dict[str, str] = {
    "2": "#8B0000",  # Compact mid-rise -- dark red
    "5": "#aaaaaa",  # Open mid-rise -- light grey
    "6": "#eeeeee",  # Open low-rise -- white/off-white
    "9": "#555555",  # Sparsely built -- dark grey
    "A": "#1a6b1a",  # Dense trees -- dark green
    "D": "#7dce7d",  # Low plants -- light green
    "G": "#1e6fa8",  # Water -- blue
}

# ---------------------------------------------------------------------------
# LCZ classes — parameter table.
# All LCZ parameters are aligned with the paper's LCZ parameter table.
# See paper for details on the underlying literature sources.
# ---------------------------------------------------------------------------

def _hex_to_hsv(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string (e.g. "#8B0000") to an OpenCV-style HSV
    tuple with H in [0, 179] and S, V in [0, 255].

    Args:
        hex_color: Hex color string, with or without a leading "#".

    Returns:
        A tuple (h, s, v) of integers scaled to OpenCV's HSV convention.
    """
    hex_color = hex_color.lstrip("#")
    # Split the hex string into R, G, B components and normalize to [0, 1]
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    # Rescale from colorsys's [0, 1] range to OpenCV's HSV ranges
    return (round(h * 179), round(s * 255), round(v * 255))


def _lcz(name, rel_temp, pop_density, cooling, height, impervious, svf, color_hex):
    """Build a single LCZ class parameter dictionary, deriving the HSV
    representation of the class's reference tile color automatically.

    Args:
        name: Human-readable LCZ class name.
        rel_temp: Relative temperature offset (°C) versus the baseline LCZ.
        pop_density: Population density (persons/km^2) for this class.
        cooling: Cooling effect (°C) this class exerts on neighboring cells.
        height: Typical building/vegetation height (m).
        impervious: Fraction of impervious (sealed) surface [0-1].
        svf: Sky view factor [0-1].
        color_hex: Reference tile color as a hex string.

    Returns:
        A dict containing all LCZ parameters plus the derived HSV values.
    """
    h, s, v = _hex_to_hsv(color_hex)
    return {
        "name": name,
        "rel_temp": rel_temp,
        "pop_density": pop_density,
        "cooling": cooling,
        "height": height,
        "impervious": impervious,
        "svf": svf,
        "color_hex": color_hex,
        "color_hsv_h": h,
        "color_hsv_s": s,
        "color_hsv_v": v,
    }


# fmt: off
LCZ_CLASSES: dict[str, dict] = {
    "2": _lcz("Compact mid-rise",  +2.60, 17150, 0.0,  17.5, 0.40, 0.450, "#8B0000"),
    "5": _lcz("Open mid-rise",     +1.25, 14550, 0.0,  17.5, 0.40, 0.650, "#FA8072"),
    "6": _lcz("Open low-rise",     +1.10, 6350,  0.0,  6.5,  0.35, 0.750, "#FF8C00"),
    "9": _lcz("Sparsely built",     0.00, 4350,  0.0,  6.5,  0.10, 0.800, "#FFD700"),
    "A": _lcz("Dense trees",       -5.80, 0,     -1.5, 16.5, 0.05, 0.200, "#006400"),
    "D": _lcz("Low plants",        -5.20, 0,     -1.0, 0.5,  0.05, 0.900, "#7CFC00"),
    "G": _lcz("Water",             -3.95, 0,     -0.5, 0.0,  0.05, 0.900, "#1E90FF"),
}
# fmt: on

LCZ_BASELINE_ID: str = "9"  # Reference LCZ class for relative temperature calculations

# ---------------------------------------------------------------------------
# KPI reachable ranges — used to scale/normalize KPI displays in the GUI.
# ---------------------------------------------------------------------------
KPI_RANGES: dict = {
    "heat_exposure": {
        "min": -3.6,
        "max": +2.6,
        "target_max": +1.0,
        "unit": "°C (relative to LCZ 9 baseline)",
    },
    "mean_temp": {
        "min": -17.3,
        "max": +2.6,
        "unit": "°C (relative to LCZ 9 baseline)",
    },
    "total_population": {
        "min": 0,
        "max": 107188,
        "target_min": 50000,
        "unit": "persons",
    },
    "green_fraction": {
        "min": 0.0,
        "max": 1.0,
        "unit": "fraction [0-1]",
    },
}

# ---------------------------------------------------------------------------
# GUI theme — dark color palette used across all PyQt6 widgets.
# ---------------------------------------------------------------------------
THEME: dict[str, str] = {
    "bg":       "#0d0f0e",
    "surface":  "#1a1917",
    "surface2": "#222120",
    "border":   "#2e2d2b",
    "text":     "#f0efed",
    "muted":    "#888580",
    "red":      "#CF1820",
    "orange":   "#EC6525",
    "purple":   "#AF368C",
}
