"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: gui/app.py — CityClimate Board main window. Hosts the PyQt6
MainWindow that wires together the camera/CV detection pipeline, climate
metrics computation, the grid/heatmap views, the metrics sidebar, the
Ollama-backed chat panel, and the admin/calibration dialog. Also defines
the toolbar's planning-task timer widget and a small overlay container
helper used to stack the grid view and heatmap overlay on top of each
other.
"""
from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QFrame,
)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    BOARD_WIDTH_PX,
    BOARD_HEIGHT_PX,
    GRID_ROWS,
    GRID_COLS,
    THEME,
    COLOR_TO_LCZ,
    SCREEN_SIMULATION,
)
from climate.assessment import compute_metrics
from gui.admin_panel import AdminPanel
from gui.chat_panel import ChatPanel
from gui.grid_view import GridView
from gui.heatmap_view import HeatmapOverlay
from gui.metrics_panel import MetricsPanel
from llm.ollama_client import OllamaClient
from llm.matrix_context import MatrixContext
from llm.prompt_builder import build_full_prompt, build_system_prompt, SessionState
from tests.sample_grids import DEMO_GRIDS, SCENARIOS
from vision.grid_extractor import diff_grids
from vision.cv_live_stream import LiveStreamDetector
from vision.cv_snapshot import SnapshotDetector

logger = logging.getLogger(__name__)


_GRID_OPTIONS: list[tuple[str, str]] = [
    ("Green City",        "green_city"),
    ("Urban Heat Island", "urban_heat_island"),
    ("All Green",         "all_green"),
    ("All Built",         "all_built"),
    ("Empty",             "empty"),
]


_SCENARIO_OPTIONS: list[tuple[str, Optional[str]]] = [
    ("\u2014 Freies Spiel \u2014",          None),
    ("Szenario 1: Gr\u00fcne Mitte",       "1"),
    ("Szenario 2: Industriebalance",  "2"),
    ("Szenario 3: Dichte Stadt",      "3"),
    ("\U0001f3ed Planungsaufgabe",            "P"),
]


_PLANNING_DURATION  = 5 * 60
_PLANNING_EXTENSION = 1 * 60


def _color_matrix_to_lcz(color_matrix):
    """Translate a detected color-name matrix into an LCZ-ID matrix using
    the shared COLOR_TO_LCZ mapping."""
    return [[COLOR_TO_LCZ.get(cell, None) for cell in row] for row in color_matrix]


def _grid_hash(grid) -> str:
    """Stable hash of current grid state to detect real changes."""
    return str(tuple(tuple(row) for row in grid))


def _format_results(metrics, grid) -> str:
    """Build the rich-text (HTML) results summary shown at the end of the
    planning task, comparing the current grid's industry-tile count and
    total population against the task's target thresholds.

    Args:
        metrics: A ClimateMetrics instance for the current grid.
        grid: The current LCZ grid (used to count industry tiles).

    Returns:
        An HTML string (joined with <br>) suitable for a QLabel with
        RichText formatting enabled.
    """
    industry_count = sum(1 for row in grid for cell in row if cell == "10")
    pop   = metrics.total_population
    he    = metrics.heat_exposure
    green = metrics.green_fraction * 100
    pop_ok = "\u2705" if pop >= 70_000 else "\u274c"
    ind_ok = "\u2705" if industry_count >= 3 else "\u274c"
    lines = [
        "<b>Ergebnis der Planungsaufgabe</b>",
        f"{ind_ok} Schwerindustrie-Felder: <b>{industry_count}</b> / 3",
        f"{pop_ok} Bev\u00f6lkerungskapazit\u00e4t: <b>{pop:,.0f}</b> / 70.000",
        f"&#x1F321; W\u00e4rmebelastungsindex: <b>{he:.2f}&nbsp;\u00b0C</b>",
        f"\U0001f33f Gr\u00fcnfl\u00e4chenanteil: <b>{green:.1f}&nbsp;%</b>",
    ]
    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# Planning-Task Toolbar Widget
# ---------------------------------------------------------------------------

class PlanningTimerWidget(QWidget):
    """Toolbar widget implementing the planning-task countdown timer.

    Shows a clock icon, remaining time, and Start/Pause and Stop buttons.
    When activated (via `activate`), it runs a 5-minute countdown; on
    expiry or manual stop, it displays a results popup summarizing the
    current grid's metrics against the task's targets, optionally
    offering a one-time 1-minute extension.
    """

    task_reset = pyqtSignal()

    _DURATION  = _PLANNING_DURATION
    _EXTENSION = _PLANNING_EXTENSION

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the widget UI and initialize timer state to idle."""
        super().__init__(parent)
        self._remaining = self._DURATION
        self._running   = False
        self._extended  = False
        self._paused    = False
        self._active    = False

        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._on_tick)

        self._build_ui()
        self._set_idle_state()

    def _build_ui(self) -> None:
        """Construct and lay out the timer's child widgets (separator,
        clock icon, time label, Start/Pause and Stop buttons)."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(6)

        sep = QLabel("|")
        sep.setStyleSheet(f"color: {THEME['border']}; padding: 0 4px;")
        layout.addWidget(sep)

        clock = QLabel("\u23f1")
        clock.setStyleSheet("font-size: 16px; background: transparent;")
        layout.addWidget(clock)

        self._time_lbl = QLabel("05:00")
        self._time_lbl.setMinimumWidth(58)
        layout.addWidget(self._time_lbl)

        self._start_btn = QPushButton("\u25b6 Start")
        self._start_btn.setFixedWidth(84)
        self._start_btn.clicked.connect(self._on_start_pause)
        layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("\u25a0 Beenden")
        self._stop_btn.setFixedWidth(94)
        self._stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(self._stop_btn)

    def activate(self, metrics_ref_fn, grid_ref_fn) -> None:
        """Activate the timer for a new planning task.

        Args:
            metrics_ref_fn: Zero-arg callable returning the current
                ClimateMetrics (used when building the results popup).
            grid_ref_fn: Zero-arg callable returning the current LCZ grid.
        """
        self._metrics_fn = metrics_ref_fn
        self._grid_fn    = grid_ref_fn
        self._remaining  = self._DURATION
        self._running    = False
        self._extended   = False
        self._paused     = False
        self._active     = True
        self._tick.stop()
        self._refresh_display()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._start_btn.setText("\u25b6 Start")
        self._start_btn.setStyleSheet(
            "background: #1e6b3a; color: #c8f0d8; border: 1px solid #2e9b5a; "
            "border-radius: 8px; padding: 4px 8px; font-weight: 700;"
        )
        self._stop_btn.setStyleSheet(
            f"background: {THEME['red']}; color: #fff; border: 1px solid #a01218; "
            "border-radius: 8px; padding: 4px 8px; font-weight: 700;"
        )

    def deactivate(self) -> None:
        """Stop the countdown and reset the widget back to its idle
        (disabled) visual state."""
        self._tick.stop()
        self._active  = False
        self._running = False
        self._set_idle_state()

    def _set_idle_state(self) -> None:
        """Reset the remaining time to the full duration and disable the
        Start/Stop buttons, rendering the widget in its idle style."""
        self._remaining = self._DURATION
        m, s = divmod(self._remaining, 60)
        self._time_lbl.setText(f"{m:02d}:{s:02d}")
        self._time_lbl.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: #555553; background: transparent; padding: 0 2px;"
        )
        if hasattr(self, "_start_btn"):
            self._start_btn.setEnabled(False)
            self._start_btn.setText("\u25b6 Start")
            self._start_btn.setStyleSheet(
                "background: #1a1a18; color: #555553; border: 1px solid #333330; "
                "border-radius: 8px; padding: 4px 8px; font-weight: 700;"
            )
            self._stop_btn.setEnabled(False)
            self._stop_btn.setStyleSheet(
                "background: #1a1a18; color: #555553; border: 1px solid #333330; "
                "border-radius: 8px; padding: 4px 8px; font-weight: 700;"
            )

    def _on_start_pause(self) -> None:
        """Handle the Start/Pause button: toggles between running and
        paused states, updating the countdown timer and button styling
        accordingly."""
        if not self._active:
            return
        if not self._running:
            self._running = True
            self._paused  = False
            self._tick.start()
            self._start_btn.setText("\u23f8 Pause")
            self._start_btn.setStyleSheet(
                "background: #5a4a00; color: #ffe680; border: 1px solid #9a8200; "
                "border-radius: 8px; padding: 4px 8px; font-weight: 700;"
            )
        else:
            self._running = False
            self._paused  = True
            self._tick.stop()
            self._start_btn.setText("\u25b6 Weiter")
            self._start_btn.setStyleSheet(
                "background: #1e6b3a; color: #c8f0d8; border: 1px solid #2e9b5a; "
                "border-radius: 8px; padding: 4px 8px; font-weight: 700;"
            )

    def _on_stop(self) -> None:
        """Handle the Stop button: stops the countdown and shows the
        results popup as a manual (non-expired) finish."""
        if not self._active:
            return
        self._tick.stop()
        self._running = False
        self._show_results_popup(time_expired=False)

    def _on_tick(self) -> None:
        """QTimer callback fired every second while running: decrements
        the remaining time and shows the results popup once it reaches
        zero."""
        self._remaining -= 1
        self._refresh_display()
        if self._remaining <= 0:
            self._tick.stop()
            self._running = False
            self._show_results_popup(time_expired=True)

    def _refresh_display(self) -> None:
        """Update the time label's text and color (white -> yellow in the
        last minute -> red at zero) to reflect self._remaining."""
        m, s = divmod(max(0, self._remaining), 60)
        self._time_lbl.setText(f"{m:02d}:{s:02d}")
        if self._remaining <= 60 and self._remaining > 0:
            color = "#e6c65b"
        elif self._remaining <= 0:
            color = "#e05a5a"
        else:
            color = "#f0efed"
        self._time_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {color}; background: transparent; padding: 0 2px;"
        )

    def _show_results_popup(self, time_expired: bool) -> None:
        """Build and show the modal results dialog summarizing the task
        outcome, with an optional "+1 minute" extension button (only
        offered once, on time expiry) and a "Reset" button that ends the
        task and emits `task_reset`.

        Args:
            time_expired: True if shown because the countdown reached
                zero; False if shown because the user manually stopped.
        """
        metrics = self._metrics_fn() if hasattr(self, "_metrics_fn") else None
        grid    = self._grid_fn()    if hasattr(self, "_grid_fn")    else []

        dlg = QDialog(self.window())
        dlg.setWindowTitle("Planungsaufgabe \u2014 Ergebnis")
        dlg.setModal(True)
        dlg.setMinimumWidth(400)
        dlg.setStyleSheet(
            f"QDialog {{ background: {THEME['surface']}; color: {THEME['text']}; }}"
            f"QLabel  {{ color: {THEME['text']}; }}"
            f"QPushButton {{ background: {THEME['surface2']}; color: {THEME['text']}; "
            f"border: 1px solid {THEME['border']}; border-radius: 8px; padding: 8px 18px; }}"
            f"QPushButton:hover {{ border-color: {THEME['orange']}; }}"
        )

        lay = QVBoxLayout(dlg)
        lay.setSpacing(14)
        lay.setContentsMargins(20, 20, 20, 20)

        heading = QLabel("\u23f0 Zeit abgelaufen!" if time_expired else "\u2714 Aufgabe beendet")
        heading.setStyleSheet("font-size: 20px; font-weight: 800;")
        lay.addWidget(heading)

        if metrics is not None:
            results_lbl = QLabel(_format_results(metrics, grid))
            results_lbl.setWordWrap(True)
            results_lbl.setTextFormat(Qt.TextFormat.RichText)
            results_lbl.setStyleSheet("font-size: 13px; line-height: 1.7;")
            lay.addWidget(results_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        if time_expired and not self._extended:
            extend_btn = QPushButton("+1 Minute verl\u00e4ngern")
            extend_btn.setStyleSheet(
                "background: #1e6b3a; color: #c8f0d8; border-color: #2e9b5a;"
            )
            def _do_extend():
                self._extended  = True
                self._remaining = self._EXTENSION
                self._running   = True
                self._tick.start()
                self._start_btn.setText("\u23f8 Pause")
                self._start_btn.setStyleSheet(
                    "background: #5a4a00; color: #ffe680; border: 1px solid #9a8200; "
                    "border-radius: 8px; padding: 4px 8px; font-weight: 700;"
                )
                self._refresh_display()
                dlg.accept()
            extend_btn.clicked.connect(_do_extend)
            btn_row.addWidget(extend_btn)

        finish_btn = QPushButton("Zur\u00fccksetzen")
        finish_btn.setStyleSheet(f"background: {THEME['red']}; color: #fff; border-color: #a01218;")
        def _do_finish():
            dlg.accept()
            self.deactivate()
            self.task_reset.emit()
        finish_btn.clicked.connect(_do_finish)
        btn_row.addWidget(finish_btn)

        lay.addLayout(btn_row)
        dlg.exec()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """The CityClimate Board's top-level application window.

    Wires together the camera/CV pipeline (live detection + snapshot),
    climate metrics computation, the grid/heatmap views, the metrics
    sidebar, the Ollama chat panel, and the admin/calibration dialog.
    Also manages the periodic update cycle (polling the live detector
    for camera-based grid changes), scenario/demo-grid loading, and the
    planning-task timer.
    """

    def __init__(
        self,
        camera=None,
        demo: bool = True,
        use_llm: bool = True,
        fullscreen: bool = False,
        initial_grid: str = "green_city",
        ssh_tunnel=None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Construct the main window, its child widgets, and start the
        background detection thread and UI update timer.

        Args:
            camera: Camera instance to use (mock, real, or RealSense). If
                None, a MockCamera is created automatically.
            demo: Whether to run in demo mode (no live camera polling in
                the update cycle; grid changes come from the mock camera
                / manual edits / scenarios instead).
            use_llm: Whether to enable the Ollama-backed chat/analysis
                features.
            fullscreen: Whether to start the window in fullscreen mode.
            initial_grid: Key into DEMO_GRIDS to load on startup when in
                demo mode.
            ssh_tunnel: Optional SSH tunnel object (stopped on window
                close, if provided).
            parent: Optional parent widget.
        """
        super().__init__(parent)

        self._demo        = demo
        self._use_llm     = use_llm
        self._heatmap_on  = False
        self._recent_changes: list[dict] = []
        self._last_metrics = None
        self._ssh_tunnel  = ssh_tunnel
        self._session     = SessionState()

        self._current_grid: list[list[Optional[str]]] = [
            [None] * GRID_COLS for _ in range(GRID_ROWS)
        ]
        self._prev_grid = copy.deepcopy(self._current_grid)
        self._last_grid_hash: str = _grid_hash(self._current_grid)

        if camera is not None:
            self._camera = camera
        else:
            from camera.mock_camera import MockCamera
            self._camera = MockCamera(grid_name=initial_grid)

        cam_idx = getattr(self._camera, '_device_index', 0)

        frame_provider_fn = (
            self._camera.get_frame
            if not self._demo and hasattr(self._camera, 'get_frame')
            else None
        )

        self._live_detector = LiveStreamDetector(
            on_matrix_ready=self._on_live_matrix_ready,
            demo=self._demo,
            camera_index=cam_idx,
            frame_provider=frame_provider_fn,
        )
        self._snapshot_detector = SnapshotDetector(
            demo=self._demo,
            camera_index=cam_idx,
        )
        self._ollama = OllamaClient(self)

        self._apply_stylesheet()
        self._build_toolbar()
        self._build_central()
        self._connect_signals()

        self._chat_panel.set_state_provider(self._get_board_state)

        if self._demo:
            self._load_demo_grid(initial_grid)

        self._live_detector.start()

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._update_cycle)
        self._timer.start()

        self.setWindowTitle("CityClimate Board")
        self.resize(1280, 800)
        if fullscreen:
            self.showFullScreen()

        QShortcut(QKeySequence("F11"), self).activated.connect(self._toggle_fullscreen)

        if self._use_llm:
            self._ollama.check_connection()
            self._ollama_check_timer = QTimer(self)
            self._ollama_check_timer.setInterval(5000)
            self._ollama_check_timer.timeout.connect(self._ollama.check_connection)
            self._ollama_check_timer.start()

    def _get_board_state(self):
        """Return (grid, metrics, source_label) for the chat panel's
        state provider, recomputing metrics if none are cached yet."""
        metrics = self._last_metrics or compute_metrics(self._current_grid)
        return self._current_grid, metrics, "demo" if self._demo else "live"

    def _activate_scenario(self, scenario_id: Optional[str]) -> None:
        """Load a predefined scenario's starting grid and reset session
        state, or return to free-play mode if scenario_id is None.

        Args:
            scenario_id: Key into SCENARIOS, or None to deactivate any
                active scenario and reset to free play.
        """
        self._planning_widget.deactivate()

        if scenario_id is None:
            self._session = SessionState(heatmap_active=self._heatmap_on)
            return

        scenario = SCENARIOS.get(scenario_id)
        if scenario is None:
            self._session = SessionState(heatmap_active=self._heatmap_on)
            return

        self._session = SessionState(
            scenario_id=scenario_id,
            scenario_name=scenario["name"],
            heatmap_active=self._heatmap_on,
        )

        start_grid = copy.deepcopy(scenario["start_grid"])
        start_grid = [row[:GRID_COLS] for row in start_grid[:GRID_ROWS]]
        for r in start_grid:
            while len(r) < GRID_COLS: r.append(None)
        while len(start_grid) < GRID_ROWS:
            start_grid.append([None] * GRID_COLS)

        if hasattr(self._camera, "set_grid"):
            self._camera.set_grid(start_grid)
        self._current_grid = start_grid
        self._prev_grid    = copy.deepcopy(start_grid)
        self._last_grid_hash = _grid_hash(start_grid)
        self._recent_changes = []

        metrics = compute_metrics(self._current_grid)
        self._last_metrics = metrics
        self._grid_view.update_grid(self._current_grid)
        self._heatmap_overlay.update_heatmap(metrics.temp_grid)
        self._metrics_panel.update_metrics(metrics, self._current_grid)

        if hasattr(self, "_chat_panel"):
            intro = f"\U0001f3ed **{scenario['name']}** gestartet\n{scenario.get('description', '')}"
            self._chat_panel.add_user_message(intro)

        self._planning_widget.activate(
            metrics_ref_fn=lambda: self._last_metrics or compute_metrics(self._current_grid),
            grid_ref_fn=lambda: self._current_grid,
        )

    def _on_live_matrix_ready(self, color_matrix):
        """Callback invoked by LiveStreamDetector once a new stable
        color matrix has been detected from the camera.

        Args:
            color_matrix: The stabilized color-name matrix from the
                detector.
        """
        lcz_grid = _color_matrix_to_lcz(color_matrix)
        new_hash = _grid_hash(lcz_grid)
        if new_hash == self._last_grid_hash:
            # Board unchanged — nothing new to apply/send/reset.
            # Prevents a self re-fire loop (reset -> re-stabilizes to the
            # same state -> fires again -> reset -> ...).
            return
        if hasattr(self._camera, "set_grid"):
            self._camera.set_grid(lcz_grid)
        self._apply_grid(lcz_grid)
        self._live_detector.reset()

    def _apply_stylesheet(self) -> None:
        """Apply the dark THEME-based QSS stylesheet to the main window
        and all its child widgets."""
        t = THEME
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: {t['bg']}; color: {t['text']};
                font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 13px;
            }}
            QToolBar {{
                background: {t['surface']}; border-bottom: 1px solid {t['border']};
                spacing: 8px; padding: 4px 12px;
            }}
            QToolBar QLabel {{ color: {t['text']}; font-size: 15px; font-weight: 700; }}
            QPushButton {{
                background: {t['surface2']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 8px;
                padding: 6px 14px; font-weight: 600;
            }}
            QPushButton:hover  {{ border-color: {t['orange']}; }}
            QPushButton:checked {{ background: {t['orange']}; color: #fff; border-color: {t['orange']}; }}
            QComboBox {{
                background: {t['surface2']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 8px; padding: 5px 10px;
            }}
            QComboBox QAbstractItemView {{
                background: {t['surface']}; color: {t['text']};
                selection-background-color: {t['surface2']}; border: 1px solid {t['border']};
            }}
            QSplitter::handle {{ background: {t['border']}; width: 2px; }}
            """
        )

    def _build_toolbar(self) -> None:
        """Construct the top toolbar: TH Köln logo, title, heatmap
        toggle, snapshot button, scenario/grid selectors, camera status
        label, planning timer, and the admin button."""
        bar = QToolBar("Main", self)
        bar.setMovable(False)
        self.addToolBar(bar)

        logo_lbl = QLabel()
        logo_path = Path(__file__).parent.parent / "assets" / "thkoeln_logo.png"
        if logo_path.exists():
            pm = QPixmap(str(logo_path)).scaledToHeight(36, Qt.TransformationMode.SmoothTransformation)
            logo_lbl.setPixmap(pm)
        else:
            logo_lbl.setText("TH K\u00f6ln")
            logo_lbl.setStyleSheet(
                "color:#fff; background:#C8102E; font-weight:800; font-size:13px; padding:4px 8px; border-radius:4px;"
            )
        logo_lbl.setContentsMargins(0, 0, 10, 0)
        bar.addWidget(logo_lbl)

        sep = QLabel("|")
        sep.setStyleSheet(f"color:{THEME['border']}; padding:0 4px;")
        bar.addWidget(sep)

        title = QLabel("CityClimate Board")
        title.setStyleSheet(
            f"color:{THEME['red']}; font-size:16px; font-weight:800; letter-spacing:1px; padding-right:20px;"
        )
        bar.addWidget(title)

        self._heatmap_btn = QPushButton("\U0001f321 Heatmap")
        self._heatmap_btn.setCheckable(True)
        self._heatmap_btn.setFixedWidth(110)
        self._heatmap_btn.clicked.connect(self._toggle_heatmap)
        bar.addWidget(self._heatmap_btn)

        self._snapshot_btn = QPushButton("\U0001f4f8 Snapshot")
        self._snapshot_btn.setFixedWidth(110)
        self._snapshot_btn.clicked.connect(self._on_snapshot)
        bar.addWidget(self._snapshot_btn)

        scenario_label = QLabel("Szenario:")
        scenario_label.setStyleSheet(f"color:{THEME['muted']}; padding-left:12px;")
        bar.addWidget(scenario_label)

        self._scenario_combo = QComboBox()
        for label, _ in _SCENARIO_OPTIONS:
            self._scenario_combo.addItem(label)
        self._scenario_combo.setFixedWidth(240)
        self._scenario_combo.currentIndexChanged.connect(self._on_scenario_combo_changed)
        bar.addWidget(self._scenario_combo)

        if self._demo:
            grid_label = QLabel("Grid:")
            grid_label.setStyleSheet(f"color:{THEME['muted']}; padding-left:12px;")
            bar.addWidget(grid_label)
            self._grid_combo = QComboBox()
            for label, _ in _GRID_OPTIONS:
                self._grid_combo.addItem(label)
            self._grid_combo.setFixedWidth(180)
            self._grid_combo.currentIndexChanged.connect(self._on_grid_combo_changed)
            bar.addWidget(self._grid_combo)
        else:
            self._cam_status_label = QLabel()
            self._cam_status_label.setStyleSheet("padding-left:12px; font-size:12px;")
            self._update_cam_status_label()
            bar.addWidget(self._cam_status_label)

        self._planning_widget = PlanningTimerWidget(self)
        self._planning_widget.task_reset.connect(self._on_planning_reset)
        bar.addWidget(self._planning_widget)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)

        admin_btn = QPushButton("\u2699 Admin")
        admin_btn.setFixedWidth(90)
        admin_btn.clicked.connect(self._open_admin)
        bar.addWidget(admin_btn)

    def _build_central(self) -> None:
        """Construct the central widget: a horizontal splitter with the
        grid view + heatmap overlay on the left, and a vertical splitter
        (metrics panel over chat panel) on the right."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        self._left_container = QWidget()
        self._left_container.setMinimumWidth(400)
        left_layout = QVBoxLayout(self._left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._grid_view = GridView()
        self._grid_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._grid_stack = _OverlayContainer(self._left_container)
        self._grid_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._heatmap_overlay = HeatmapOverlay(self._grid_stack)
        self._grid_stack.set_widgets(self._grid_view, self._heatmap_overlay)
        left_layout.addWidget(self._grid_stack)

        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setHandleWidth(2)

        self._metrics_panel = MetricsPanel()
        self._chat_panel    = ChatPanel()

        self._right_splitter.addWidget(self._metrics_panel)
        self._right_splitter.addWidget(self._chat_panel)
        self._right_splitter.setSizes([450, 550])

        splitter.addWidget(self._left_container)
        splitter.addWidget(self._right_splitter)
        splitter.setSizes([768, 512])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        """Wire up cross-widget Qt signal/slot connections (grid edits,
        chat requests, Ollama streaming/status callbacks)."""
        self._grid_view.cell_changed.connect(self._on_cell_changed)
        self._chat_panel.send_requested.connect(self._on_chat_send)
        self._chat_panel.analyze_requested.connect(self._on_analyze_requested)
        self._ollama.token_received.connect(self._chat_panel.add_ai_token)
        self._ollama.response_complete.connect(self._on_llm_response_complete)
        self._ollama.error_occurred.connect(self._on_llm_error)
        self._ollama.connection_status.connect(self._chat_panel.set_connection_status)

    def _update_cycle(self) -> None:
        """Periodic QTimer callback (every 500 ms). In live-camera mode,
        polls the live detector's latest stabilized matrix, and if it
        represents a genuine change from the previous grid, applies it
        and records the diff. No-op in demo mode (grid changes there
        come from manual edits, scenarios, or the mock camera directly).
        """
        try:
            self._update_cam_status_label()

            if self._demo:
                return

            color_matrix = self._live_detector.last_matrix
            if color_matrix is None:
                return
            new_grid = _color_matrix_to_lcz(color_matrix)
            if len(new_grid) != GRID_ROWS or any(len(r) != GRID_COLS for r in new_grid):
                return
            new_hash = _grid_hash(new_grid)
            if new_hash != self._last_grid_hash:
                changes = diff_grids(self._prev_grid, new_grid)
                if changes:
                    self._recent_changes = changes[-10:]
                    self._prev_grid = copy.deepcopy(new_grid)
                    self._last_grid_hash = new_hash
                    self._apply_grid(new_grid)
        except Exception as exc:
            logger.warning("update_cycle error: %s", exc)

    def _on_snapshot(self) -> None:
        """Handle the Snapshot button: captures a single frame via the
        SnapshotDetector (using a synthetic frame derived from the
        current grid when in screen-simulation demo mode) and applies
        the detected grid."""
        try:
            self._snapshot_btn.setEnabled(False)
            self._snapshot_btn.setText("\u23f3 Snapshot\u2026")
            sim_matrix = None
            if SCREEN_SIMULATION and self._demo:
                lcz_to_color = {v: k for k, v in COLOR_TO_LCZ.items() if v is not None}
                sim_matrix = [[lcz_to_color.get(cell, "empty") for cell in row] for row in self._current_grid]
            color_matrix, _ = self._snapshot_detector.take_snapshot(sim_matrix=sim_matrix)
            if color_matrix is None:
                return
            self._apply_grid(_color_matrix_to_lcz(color_matrix))
        except Exception as exc:
            logger.warning("Snapshot error: %s", exc)
        finally:
            self._snapshot_btn.setEnabled(True)
            self._snapshot_btn.setText("\U0001f4f8 Snapshot")

    def _apply_grid(self, new_grid, record_move: bool = False) -> None:
        """Apply a new LCZ grid: recompute metrics, refresh the grid
        view, heatmap overlay, and metrics panel, optionally record the
        move in the active scenario session, and (in live, LLM-enabled
        mode) request a short automatic analysis from Ollama.

        Args:
            new_grid: The new LCZ grid to apply.
            record_move: If True and a scenario is active, records this
                move's metrics into the session history.
        """
        self._current_grid = new_grid
        self._last_grid_hash = _grid_hash(new_grid)
        metrics = compute_metrics(self._current_grid)
        self._last_metrics = metrics
        if record_move and self._session.scenario_id:
            self._session.record_move(metrics)
        self._grid_view.update_grid(self._current_grid)
        self._heatmap_overlay.update_heatmap(metrics.temp_grid)
        self._metrics_panel.update_metrics(metrics, self._current_grid)
        if self._use_llm and not self._demo:
            ctx = MatrixContext(grid=self._current_grid, metrics=metrics, timestamp=time.time(), source="live")
            context_str = build_full_prompt(ctx, session=self._session)
            self._chat_panel.start_ai_message()
            self._ollama.send_message(
                "Grid aktualisiert \u2014 kurze Analyse bitte.",
                context_str=context_str,
                history=[],
            )

    def _update_cam_status_label(self) -> None:
        """Refresh the camera status label (green "active" / red "no
        camera") if it exists (only present in non-demo mode)."""
        if not hasattr(self, "_cam_status_label"):
            return
        if hasattr(self._camera, "is_open") and self._camera.is_open():
            self._cam_status_label.setText("\U0001f7e2 Kamera aktiv")
            self._cam_status_label.setStyleSheet("color:#44d17a; padding-left:12px; font-size:12px;")
        else:
            self._cam_status_label.setText("\U0001f534 Keine Kamera")
            self._cam_status_label.setStyleSheet("color:#ff6b6b; padding-left:12px; font-size:12px;")

    def _load_demo_grid(self, grid_name: str) -> None:
        """Load a predefined demo grid (from DEMO_GRIDS), normalize it to
        GRID_ROWS x GRID_COLS, push it into the mock camera if present,
        and refresh all dependent views/metrics.

        Args:
            grid_name: Key into DEMO_GRIDS selecting which demo grid to
                load.
        """
        grid = copy.deepcopy(DEMO_GRIDS[grid_name])
        grid = [row[:GRID_COLS] for row in grid[:GRID_ROWS]]
        for r in grid:
            while len(r) < GRID_COLS: r.append(None)
        while len(grid) < GRID_ROWS:
            grid.append([None] * GRID_COLS)
        if hasattr(self._camera, "set_grid"):
            self._camera.set_grid(grid)
        self._current_grid = grid
        self._prev_grid    = copy.deepcopy(grid)
        self._last_grid_hash = _grid_hash(grid)
        self._recent_changes = []
        metrics = compute_metrics(self._current_grid)
        self._last_metrics = metrics
        self._grid_view.update_grid(self._current_grid)
        self._heatmap_overlay.update_heatmap(metrics.temp_grid)
        self._metrics_panel.update_metrics(metrics, self._current_grid)

    def _on_grid_combo_changed(self, index: int) -> None:
        """Handle the demo-grid selector: load the newly selected demo
        grid."""
        _, grid_key = _GRID_OPTIONS[index]
        self._load_demo_grid(grid_key)

    def _on_scenario_combo_changed(self, index: int) -> None:
        """Handle the scenario selector: activate the newly selected
        scenario (or return to free play)."""
        _, scenario_id = _SCENARIO_OPTIONS[index]
        self._activate_scenario(scenario_id)

    def _on_planning_reset(self) -> None:
        """Handle the planning timer's task_reset signal: reset the
        scenario selector to free play and deactivate any active
        scenario."""
        self._scenario_combo.setCurrentIndex(0)
        self._activate_scenario(None)

    def _toggle_heatmap(self, checked: bool) -> None:
        """Toggle the temperature heatmap overlay's visibility and update
        the toggle button's label and session state."""
        self._heatmap_on = checked
        self._heatmap_overlay.set_visible(checked)
        self._session.heatmap_active = checked
        self._heatmap_btn.setText("\U0001f321 Heatmap \u2713" if checked else "\U0001f321 Heatmap")

    def _toggle_fullscreen(self) -> None:
        """Toggle between fullscreen and normal window mode (bound to
        F11)."""
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _open_admin(self) -> None:
        """Open the modal AdminPanel dialog for calibration/settings, then
        reload the camera's calibration and the CV color ranges
        regardless of whether the dialog was accepted or rejected."""
        frame_provider = getattr(self._camera, "get_frame", None)
        dlg = AdminPanel(parent=self, frame_provider=frame_provider,
                         camera=self._camera if not self._demo else None)
        dlg.exec()  # always reload after close, regardless of accept/reject
        if hasattr(self._camera, "reload_calibration"):
            self._camera.reload_calibration()
        self._reload_color_ranges()

    def _reload_color_ranges(self) -> None:
        """Reload color_ranges.json into COLOR_RANGES and reset the stable-matrix
        buffer so the detector immediately uses the new/disabled colors.
        """
        try:
            from vision.cv_live_stream import _load_color_ranges_from_file, COLOR_RANGES, DISABLED_COLORS
            _load_color_ranges_from_file()
            active = sorted(COLOR_RANGES.keys())
            disabled = sorted(DISABLED_COLORS)
            logger.info(
                "COLOR_RANGES reloaded: active=%s  disabled=%s",
                active, disabled,
            )
        except Exception as exc:
            logger.warning("Could not reload color ranges: %s", exc)

        # Clear the stable-matrix vote buffer so stale frames don’t bleed through
        try:
            self._live_detector.reset(clear_last_matrix=True)
        except Exception:
            pass

    def _on_cell_changed(self, row: int, col: int, lcz_id: str) -> None:
        """Handle a manual grid-cell edit from GridView: push the change
        to the mock camera (if present), update the current/previous grid
        state, recompute metrics, record the move in the active session
        (if any), and refresh all dependent views.

        Args:
            row: Row index of the edited cell.
            col: Column index of the edited cell.
            lcz_id: New LCZ ID for the cell, or a falsy value to clear it.
        """
        value = lcz_id if lcz_id else None
        if hasattr(self._camera, "update_cell"):
            self._camera.update_cell(row, col, value)
        self._current_grid[row][col] = value
        self._prev_grid[row][col]    = value
        self._last_grid_hash = _grid_hash(self._current_grid)
        metrics = compute_metrics(self._current_grid)
        self._last_metrics = metrics
        if self._session.scenario_id:
            self._session.record_move(metrics)
        self._session.last_suggestion = None
        self._grid_view.update_grid(self._current_grid)
        self._heatmap_overlay.update_heatmap(metrics.temp_grid)
        self._metrics_panel.update_metrics(metrics, self._current_grid)

    def _on_chat_send(self, user_text: str, history: list) -> None:
        """Handle a user-submitted chat message: build the current
        board-state context and forward the message to Ollama (or show a
        disabled-LLM notice if --no-llm was used).

        Args:
            user_text: The user's chat message text.
            history: The chat panel's prior message history.
        """
        if not self._use_llm:
            self._chat_panel.start_ai_message()
            self._chat_panel.add_ai_token("[LLM disabled \u2014 start without --no-llm to enable Ollama.]")
            self._chat_panel.finish_ai_message()
            return
        metrics = self._last_metrics or compute_metrics(self._current_grid)
        ctx = MatrixContext(
            grid=self._current_grid, metrics=metrics,
            timestamp=time.time(), source="demo" if self._demo else "live"
        )
        context_str = build_full_prompt(ctx, session=self._session, user_message=user_text)
        self._chat_panel.start_ai_message()
        self._ollama.send_message(
            user_text,
            context_str=context_str,
            history=history,
        )

    def _on_analyze_requested(self) -> None:
        """Handle the chat panel's "analyze current configuration"
        button: sends an automatic analysis request to Ollama with the
        current board-state context."""
        if not self._use_llm:
            self._chat_panel.start_ai_message()
            self._chat_panel.add_ai_token("[LLM disabled]")
            self._chat_panel.finish_ai_message()
            return
        metrics = self._last_metrics or compute_metrics(self._current_grid)
        ctx = MatrixContext(grid=self._current_grid, metrics=metrics, timestamp=time.time(),
                            source="demo" if self._demo else "live")
        context_str = build_full_prompt(ctx, session=self._session)
        self._chat_panel.start_ai_message()
        self._ollama.send_message(
            "Analysiere die aktuelle Konfiguration.",
            context_str=context_str,
            history=list(self._chat_panel._history),
        )

    def _on_llm_response_complete(self, full_text: str) -> None:
        """Handle Ollama's response completion: finalize the chat
        message, then scan the response text for known LCZ zone keywords
        (in German/English) to remember the model's most recent zone
        suggestion for follow-up context.

        Args:
            full_text: The complete assistant response text.
        """
        self._chat_panel.finish_ai_message(full_text)
        zone_keywords = {
            "A": ["dichter baumbestand", "dense trees", "zone a"],
            "B": ["aufgelockerter baumbestand", "scattered trees", "zone b"],
            "D": ["niedrige vegetation", "low plants", "zone d"],
            "G": ["wasserfläche", "water", "zone g"],
            "5": ["offene mittelhochbebauung", "open mid-rise", "zone 5"],
            "6": ["offene niedrigbebauung", "open low-rise", "zone 6"],
            "9": ["lockere bebauung", "sparsely built", "zone 9"],
        }
        lower = full_text.lower()
        for zone_id, keywords in zone_keywords.items():
            if any(kw in lower for kw in keywords):
                self._session.last_suggestion = zone_id
                break

    def _on_llm_error(self, msg: str) -> None:
        """Handle an Ollama error: log it and finalize the (empty) chat
        message so the UI doesn't remain stuck in a "typing" state."""
        logger.warning("LLM error: %s", msg)
        self._chat_panel.finish_ai_message()

    def closeEvent(self, event) -> None:
        """Qt close-event handler: stop all timers/threads, cancel any
        in-flight Ollama request, release the camera, stop the SSH
        tunnel (if any), and deactivate the planning widget before
        allowing the window to close."""
        self._timer.stop()
        if hasattr(self, "_ollama_check_timer"):
            self._ollama_check_timer.stop()
        self._live_detector.stop()
        if hasattr(self._snapshot_detector, "close"):
            self._snapshot_detector.close()
        self._ollama.cancel()
        if hasattr(self._camera, "release"):
            self._camera.release()
        if self._ssh_tunnel is not None:
            self._ssh_tunnel.stop()
        self._planning_widget.deactivate()
        event.accept()


class _OverlayContainer(QWidget):
    """Simple container that stacks two child widgets (base + overlay) on
    top of each other, both resized to fill the container's full size.
    Used to overlay the semi-transparent heatmap on top of the grid
    view."""

    def __init__(self, parent=None):
        """Initialize the container with no widgets assigned yet."""
        super().__init__(parent)
        self._base = self._overlay = None

    def set_widgets(self, base, overlay):
        """Assign the base and overlay widgets and lay them out to fill
        the container.

        Args:
            base: The bottom widget (e.g. GridView).
            overlay: The top widget (e.g. HeatmapOverlay), drawn on top
                of `base`.
        """
        self._base = base
        self._overlay = overlay
        base.setParent(self)
        overlay.setParent(self)
        self._relayout()

    def resizeEvent(self, event):
        """Qt resize-event handler: re-fit both child widgets to the new
        container size."""
        self._relayout()
        super().resizeEvent(event)

    def _relayout(self):
        """Resize both the base and overlay widgets to exactly fill the
        container's current geometry."""
        for w in (self._base, self._overlay):
            if w:
                w.setGeometry(0, 0, self.width(), self.height())
