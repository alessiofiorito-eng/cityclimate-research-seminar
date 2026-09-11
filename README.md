# CityClimate Board

An interactive tangible interface for real-time analysis of urban climate scenarios.
Colored building blocks on a physical board are detected via camera, mapped to Local Climate Zone (LCZ) classes, and evaluated thermally.

## Prerequisites

- Python 3.11+ (developed and tested against this range)
- Windows, Linux, or macOS — camera backends are auto-selected per platform
- Optional: an Intel RealSense D435i (or compatible) for the recommended input mode

## Installation

```bash
git clone https://github.com/alessiofiorito-eng/Forschungsseminar.git
cd Forschungsseminar
```

Create and activate a virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script, allow it once for the current session:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Then install the dependencies:

```bash
cd cityclimate
pip install -r requirements.txt
```

## Quick Start

```bash
# Demo mode — no camera, no Ollama required
python main.py --demo

# With a real webcam
python main.py --camera

# With an Intel RealSense camera (recommended, if hardware is available)
python main.py --realsense
```

Press F11 to toggle fullscreen.

### Input modes

- **`--demo`** — no physical hardware required. A software mock camera
  (`camera/mock_camera.py`) renders a predefined LCZ grid directly as a
  synthetic frame, so the full GUI, climate model, and chat can be explored
  without a board or camera on hand.
- **`--camera`** — uses a standard webcam via `cv2.VideoCapture`
  (`camera/real_camera.py`). Requires the board to be calibrated once via the
  Admin Panel (see below).
- **`--realsense`** — uses an Intel RealSense RGB stream via `pyrealsense2`
  (`camera/realsense_camera.py`), the setup this project was built and tested
  with (see [`docs/experimental_setup.md`](docs/experimental_setup.md) for the
  exact hardware and configuration used). Requires `pip install pyrealsense2`
  separately if it's not already present in your environment.

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--demo` | on (implicit) | Run without camera/board, using a synthetic demo grid |
| `--camera` | off | Use a standard webcam via `cv2.VideoCapture` |
| `--realsense` | off | Use an Intel RealSense RGB stream via `pyrealsense2` |
| `--camera-index N` | `0` | `cv2.VideoCapture` device index (only with `--camera`) |
| `--grid {baseline,urban_heat_island,green_city}` | `green_city` | Initial demo grid (only in `--demo` mode) |
| `--no-llm` | off | Disable Ollama integration (chat panel stays visible but inactive) |
| `--fullscreen` | off | Start in fullscreen mode (toggle anytime with F11) |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | `INFO` | Logging verbosity |

## Ollama Integration (optional)

The LLM chat connects to `http://localhost:11434` by default (see `llm/llm_config.yaml`).

**Local Ollama:** Start `ollama serve`, then build the model once:
```bash
ollama create climate-analyst -f cityclimate/llm/modelfile
```

**Ollama on a remote server (e.g. via SSH):**
1. Start the app, open the Admin Panel → "Ollama" tab.
2. Enter host, port, and username, then click "Open SSH Tunnel".
3. Enter the password in the terminal window that opens.
4. Once the tunnel is up, the chat connects automatically (no restart needed).

If your model name or host differs, adjust `llm/llm_config.yaml` accordingly.

## CV Pipeline

Color detection (`vision/cv_live_stream.py`) recognizes 7 colors (blue, dark gray, dark green,
light gray, light green, red, white) via HSV masking with majority-vote stabilization,
and maps them to the corresponding LCZ types.

In the **Admin Panel → "Color Calibration" tab**, HSV ranges can be recalibrated live via
snapshot + click-to-assign.

## Features

- **Grid view**: 5×5 LCZ cells with color coding and hover tooltips
- **Heatmap**: temperature overlay (RdYlGn) pixel-aligned to the grid cells
- **Metrics panel**: heat exposure, mean temperature, population, green fraction
- **LLM chat**: Ollama integration with streaming (SSH tunnel support)
- **Admin panel**: calibration, camera selection, Ollama configuration, HSV tuning

## Documentation

All modules and scripts in this repository are fully documented in code
(module/class/function docstrings and inline comments) — see the source
directly for implementation-level detail. For system-level documentation,
refer to:

- [`docs/architecture.md`](docs/architecture.md) — the full system architecture:
  module overview, data flow through the vision/climate/GUI/LLM pipeline,
  configuration reference, and known limitations. Start here for an overview
  of how the codebase fits together.
- [`docs/experimental_setup.md`](docs/experimental_setup.md) — the physical
  experimental setup: camera hardware and positioning, lighting requirements,
  the board-and-color calibration procedure, the tile-to-LCZ color scheme, and
  the compute environment used for the local LLM.
- [`docs/flowcharts.md`](docs/flowcharts.md) — process flowcharts for
  application startup, the calibration workflows, the chat/LLM request
  decision flow, and the scenario/planning-task lifecycle.
- [`cad/`](cad/) — the CAD models used for the physical grid base plate and
  the individual LCZ tiles.

## Troubleshooting

- **`RuntimeError: pyrealsense2 ist nicht installiert`** — the `pyrealsense2`
  package is not installed in your active environment. Run
  `pip install pyrealsense2`, or use `--demo` / `--camera` instead if no
  RealSense camera is available.
- **PowerShell: "cannot be loaded because running scripts is disabled"** —
  run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` once
  per session, then retry activating the virtual environment.
- **Chat panel stuck offline** — Ollama is not reachable at the configured
  `OLLAMA_BASE_URL`. Verify `ollama serve` is running locally, or that the SSH
  tunnel (see above) is open and forwarding the correct port.
- **`ERROR: Could not open requirements file`** — make sure you're inside the
  `cityclimate/` directory before running `pip install -r requirements.txt`.

## Team

Research Seminar, Summer Term 2026, TH Köln

- Alessio Fiorito
- Bakir Ahmetbegovic
- Segmen Bagcivan
