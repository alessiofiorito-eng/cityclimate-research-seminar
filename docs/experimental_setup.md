# CityClimate Board — Experimental Setup

Research Seminar, Summer Term 2026 · TH Köln
Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

This document describes the physical setup of the CityClimate Board prototype:
camera hardware, positioning, lighting, calibration, and the compute
environment used for the local LLM. For the software architecture, see
[`architecture.md`](./architecture.md).

---

## 1. Camera Hardware

| Property | Value |
|---|---|
| Model | Intel RealSense Depth Camera D435i |
| Product code | 82635D435IDK5P |
| Stream used | RGB color stream (depth stream currently unused) |
| Interface | `pyrealsense2`, see `camera/realsense_camera.py` |
| Distance to grid | 25 cm (perpendicular) |

The camera is driven via `python main.py --realsense`. For details on the
API-compatible camera abstraction (`RealSenseCamera` as a drop-in replacement
for `RealCamera`), see `architecture.md`, "Module Overview" section.

## 2. Physical Positioning

- **Orientation:** the camera is mounted directly above the center of the 5×5
  LCZ grid, **with no tilt** (the optical axis is perpendicular to the grid
  plane). This minimizes the perspective distortion that the homography
  computation would otherwise need to compensate for.
- **Distance:** 25 cm between the camera lens and the grid surface.
- **Positioning goal:** the entire 5×5 grid should fill the camera frame with
  as uniform a per-cell resolution as possible, so that the later homography
  rectification (`vision/calibration.py`) has to correct for as little
  distortion as possible.

## 3. Calibration

Position/perspective calibration was performed **exclusively via the
calibration tool built into the application** (Admin Panel → "Calibration"
tab), not manually or with an external tool:

1. Open the live camera feed in the Admin Panel.
2. Click the 4 corners of the physical 5×5 grid in order: top-left →
   top-right → bottom-right → bottom-left.
3. The tool automatically computes the homography matrix
   (`cv2.findHomography`, see `vision/calibration.py::compute_homography`) and
   shows a warp preview including grid lines and the per-cell color-sampling
   regions.
4. Saving writes the matrix to `calibration.json`, which is then used by live
   detection (`vision/cv_live_stream.py`) for every frame.

**Color calibration** was likewise performed via the built-in tool (Admin
Panel → "Color Calibration" tab): the board was frozen via a snapshot,
individual tiles were clicked and assigned to the correct LCZ color, and the
tool automatically computed updated HSV ranges from these samples
(`compute_ranges_from_samples`, see `architecture.md`, "Vision Pipeline"
section).

## 4. Lighting

Reliable color detection by the CV pipeline required **uniform, direct
lighting** of the grid — this was essential. Uneven lighting (shadows,
reflections, backlight) shifts values in HSV color space and causes
misclassification of individual cells (see `tools/hsv_logger.py` for
diagnosing such effects). In practice, this means:

- The light source should be as diffuse as possible and positioned overhead,
  without hard cast shadows from the tiles or the camera mount falling onto
  the grid.
- Avoid one-sided lighting that creates a brightness/color cast across the
  grid (e.g. a window on one side without compensation).
- Lighting conditions should be kept as consistent as possible between
  calibration and actual use, since the HSV ranges are tuned to the
  calibration conditions.

## 5. Color Scheme (Tile ↔ LCZ)

The physical tiles are color-coded and mapped to their respective LCZ class
via `config.COLOR_TO_LCZ`:

| Tile color | LCZ ID | Name |
|---|---|---|
| Red | 2 | Compact mid-rise |
| Light gray | 5 | Open mid-rise |
| White | 6 | Open low-rise |
| Dark gray | 9 | Sparsely built |
| Dark green | A | Dense trees |
| Light green | D | Low plants |
| Blue | G | Water |

For details on the underlying physical LCZ parameters (temperature,
population density, cooling effect, etc.), see `config.py` (`LCZ_CLASSES`) and
the paper.

## 6. CAD Files

The build plans for the grid base plate and the individual LCZ tiles are
located in the [`cad/`](../cad/README.md) folder.

## 7. Compute Environment / LLM Server

The local LLM does not run on the machine controlling the GUI, but on a
separate edge-compute device connected via an SSH tunnel (see
`architecture.md`, "LLM Integration" section).

| Property | Value |
|---|---|
| Hardware | ASUS GX10 |
| Hostname | `ascent-ccl` |
| Architecture | arm64 |
| Operating system | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Kernel | Linux 6.17.0-1031-nvidia |

### Model

The LLM used is **`climate-analyst`** — a model based on **Llama 3.1**, built
with Ollama from the Modelfile checked into the repository (`llm/modelfile`).
The system prompt and response behavior (German-language analysis, fixed
formatting rules) are additionally defined in `llm/prompt_builder.py`
(`SYSTEM_PROMPT`) and are combined with the current grid context on every
request.

The GUI accesses the model via a manually started SSH tunnel (Admin Panel →
"Ollama" tab) that forwards the GX10's Ollama REST port locally, so that
`OllamaClient` (see `llm/ollama_client.py`) can communicate transparently
against `http://localhost:<local port>`.
