#!/usr/bin/env python3
"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: main.py — Application entry point for the CityClimate Board.
Parses command-line arguments to select the input source (mock camera,
webcam, or Intel RealSense), constructs the appropriate camera object,
configures logging, and launches the PyQt6 GUI (MainWindow).
"""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_args() -> argparse.Namespace:
    """Define and parse the command-line interface for the application.

    Sets up a mutually exclusive group for the input source (--demo,
    --camera, --realsense) plus additional flags controlling the initial
    demo grid, LLM integration, fullscreen mode, and logging verbosity.

    Returns:
        The parsed argparse.Namespace containing all CLI options.
    """
    parser = argparse.ArgumentParser(
        prog="cityclimate",
        description="CityClimate Board — tangible urban climate planning tool",
    )

    # Only one input source may be selected at a time
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Run in demo mode (no camera, no Ollama required). Default.",
    )
    mode.add_argument(
        "--camera",
        action="store_true",
        default=False,
        help="Use real webcam via cv2.VideoCapture.",
    )
    mode.add_argument(
        "--realsense",
        action="store_true",
        default=False,
        help="Use Intel RealSense RGB stream via pyrealsense2 (recommended).",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        metavar="N",
        help="cv2.VideoCapture device index (only used with --camera, default: 0).",
    )

    parser.add_argument(
        "--grid",
        choices=["baseline", "urban_heat_island", "green_city"],
        default="green_city",
        metavar="GRID",
        help="Initial demo grid (only used in --demo mode). "
             "Choices: baseline, urban_heat_island, green_city. Default: green_city.",
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help="Disable Ollama LLM integration (chat panel visible but inactive).",
    )

    parser.add_argument(
        "--fullscreen",
        action="store_true",
        default=False,
        help="Start in fullscreen mode (toggle with F11).",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity. Default: INFO.",
    )

    return parser.parse_args()


def _grid_key(raw: str) -> str:
    """Map CLI --grid value to DEMO_GRIDS key.

    The CLI exposes "baseline" as a user-friendly alias for the
    "urban_heat_island" demo grid preset; all other values pass through
    unchanged.

    Args:
        raw: The raw --grid value as provided on the command line.

    Returns:
        The corresponding key into the DEMO_GRIDS mapping.
    """
    return "urban_heat_island" if raw == "baseline" else raw


def _build_camera(args: argparse.Namespace):
    """Instantiate and return the correct camera object based on CLI flags.

    Selects between the RealSense camera, a standard webcam, or the mock
    camera (default), depending on which mode flag was passed. For real
    hardware, logs a warning (but does not crash) if the device cannot be
    opened, allowing the app to still start in a fallback state.

    Args:
        args: Parsed CLI arguments returned by _parse_args().

    Returns:
        A camera instance implementing the shared camera interface.
    """
    if args.realsense:
        from camera.realsense_camera import RealSenseCamera
        cam = RealSenseCamera()
        if not cam.is_open():
            logging.getLogger(__name__).warning(
                "RealSense not found or pyrealsense2 unavailable — "
                "running in fallback mode."
            )
        return cam
    elif args.camera:
        from camera.real_camera import RealCamera
        cam = RealCamera(device_index=args.camera_index)
        if not cam.is_open():
            logging.getLogger(__name__).warning(
                "No webcam found — RealCamera running in fallback mode. "
                "The app will still start; calibrate via the Admin panel once a "
                "camera is connected."
            )
        return cam
    else:
        from camera.mock_camera import MockCamera
        return MockCamera(grid_name=_grid_key(args.grid))


def main() -> None:
    """Application entry point.

    Parses CLI arguments, configures logging, builds the selected camera
    backend, then creates and shows the PyQt6 MainWindow before entering
    the Qt event loop.
    """
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # The SSH tunnel is now started manually from the Admin panel ("Ollama"
    # tab). No automatic tunnel setup happens at application startup anymore.
    tunnel = None

    camera = _build_camera(args)

    # PyQt6 must be imported *after* logging is configured
    from PyQt6.QtWidgets import QApplication
    from gui.app import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("CityClimate Board")
    app.setOrganizationName("TH Köln")

    window = MainWindow(
        camera=camera,
        demo=not (args.camera or args.realsense),
        use_llm=not args.no_llm,
        fullscreen=args.fullscreen,
        initial_grid=_grid_key(args.grid),
        ssh_tunnel=tunnel,
    )
    window.show()

    exit_code = app.exec()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
