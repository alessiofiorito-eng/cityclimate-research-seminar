"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: gui/admin_panel.py — Admin panel dialog: camera selection,
board calibration (homography via 4 clicked corners), Ollama/SSH-tunnel
connection settings, and HSV color calibration for the CV detection
pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

import subprocess
import platform

import httpx

from PyQt6.QtCore import Qt, QObject, QPoint, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction, QColor, QCursor, QImage, QMouseEvent,
    QPainter, QPen, QPixmap, QFont, QIcon,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

try:
    import requests
except ImportError:
    requests = None

try:
    import paramiko  # noqa: F401
    _PARAMIKO_AVAILABLE = True
except ImportError:
    _PARAMIKO_AVAILABLE = False

import cv2


LCZ_ORDER      = ["2", "5", "6", "9", "10", "A", "B", "D", "G"]
_CORNER_LABELS = ["1: oben-links", "2: oben-rechts", "3: unten-rechts", "4: unten-links"]
_GRID_N        = 5
_SAMPLE_FRAC   = 0.40

_ALL_COLOR_NAMES = ["blue", "darkgray", "darkgreen", "lightgray", "lightgreen", "red", "white", "empty"]

_COLOR_EMOJI: dict[str, str] = {
    "blue":       "\U0001f7e6",
    "darkgray":   "\u2b1b",
    "darkgreen":  "\U0001f7e9",
    "lightgray":  "\u2591",
    "lightgreen": "\U0001f7e2",
    "red":        "\U0001f7e5",
    "white":      "\u2b1c",
    "empty":      "\u2715",
}

_COLOR_HEX: dict[str, str] = {
    "blue":       "#1e90ff",
    "darkgray":   "#555555",
    "darkgreen":  "#006400",
    "lightgray":  "#aaaaaa",
    "lightgreen": "#7cfc00",
    "red":        "#cf1820",
    "white":      "#eeeeee",
    "empty":      "#333333",
}

# Colors that can be toggled ("empty" is always active, can't be disabled)
_TOGGLEABLE_COLORS = ["blue", "darkgray", "darkgreen", "lightgray", "lightgreen", "red", "white"]

_HSV_H_MARGIN = 8
_HSV_S_MARGIN = 30
_HSV_V_MARGIN = 30


def _safe_font(size: int = 9) -> QFont:
    """Return a QFont with the given point size, clamped to a minimum of 8."""
    f = QFont()
    f.setPointSize(max(8, size))
    return f


def _make_swatch_icon(color_hex: str, size: int = 16) -> QIcon:
    """Render a small square color-swatch icon (with a thin border) for
    use in menus/lists, e.g. the per-cell color picker context menu."""
    pm = QPixmap(size, size)
    pm.fill(QColor(color_hex))
    p = QPainter(pm)
    p.setPen(QPen(QColor("#555"), 1))
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return QIcon(pm)


def _draw_grid_overlay(
    canvas: np.ndarray,
    corners: list[QPoint],
    n: int = _GRID_N,
    sample_frac: float = _SAMPLE_FRAC,
) -> None:
    """Draw the perspective-corrected grid overlay (cyan grid lines plus
    green per-cell sampling boxes) onto `canvas`, in place.

    Computes a homography from the unit n x n grid to the 4 given
    corner points (in image pixel space), then projects both the grid
    lines and the inner sampling boxes for each cell through that
    homography before drawing them.

    Args:
        canvas: BGR image to draw onto, modified in place.
        corners: Exactly 4 clicked corner points (top-left, top-right,
            bottom-right, bottom-left) in image pixel coordinates.
        n: Number of grid cells per side.
        sample_frac: Fraction of each cell's width/height used for the
            inner sampling box (the rest is margin, matching
            CELL_MARGIN in the detection pipeline).
    """
    if len(corners) != 4:
        return
    src = np.array([[0, 0], [n, 0], [n, n], [0, n]], dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array([[p.x(), p.y()] for p in corners], dtype=np.float32).reshape(-1, 1, 2)
    try:
        M, _ = cv2.findHomography(src, dst)
    except Exception:
        return
    if M is None:
        return
    for i in range(n + 1):
        for axis in [0, 1]:
            pts_src = np.array(
                [[[i, j] for j in range(n + 1)] if axis == 0
                 else [[j, i] for j in range(n + 1)]],
                dtype=np.float32,
            )
            pts_dst = cv2.perspectiveTransform(pts_src, M)[0].astype(int)
            for k in range(len(pts_dst) - 1):
                cv2.line(canvas, tuple(pts_dst[k]), tuple(pts_dst[k + 1]),
                         (0, 220, 255), 1, cv2.LINE_AA)
    margin = sample_frac / 2
    for row in range(n):
        for col in range(n):
            box_src = np.array([[
                [col + margin,     row + margin],
                [col + 1 - margin, row + margin],
                [col + 1 - margin, row + 1 - margin],
                [col + margin,     row + 1 - margin],
            ]], dtype=np.float32)
            box_dst = cv2.perspectiveTransform(box_src, M)[0].astype(int)
            cv2.polylines(canvas, [box_dst], True, (0, 255, 180), 1, cv2.LINE_AA)


class _CameraScanWorker(QThread):
    """Background thread that scans for available camera devices without
    blocking the UI, emitting `scan_done` with the list of found device
    indices once finished."""

    scan_done = pyqtSignal(list)

    def run(self) -> None:
        """Thread entry point: run the camera scan and emit the results
        (an empty list on any failure)."""
        try:
            from camera.real_camera import scan_cameras
            found = scan_cameras(max_index=6, timeout_ms=1500)
        except Exception:
            found = []
        self.scan_done.emit(found)


class _SshConnectWorker(QThread):
    """Background thread that establishes an SSH tunnel (via Paramiko)
    without blocking the UI, emitting either `connected` (tunnel object,
    status text, log text) on success or `failed` (error message) on
    failure."""

    connected = pyqtSignal(object, str, str)
    failed    = pyqtSignal(str)

    def __init__(self, cfg: dict, parent: Optional[QObject] = None) -> None:
        """Store the tunnel configuration dict (host, user, ports, etc.)
        to be used when the thread runs."""
        super().__init__(parent)
        self._cfg = cfg

    def run(self) -> None:
        """Thread entry point: attempt to open the SSH tunnel and emit
        the appropriate signal based on the outcome."""
        try:
            from utils.ssh_tunnel import _ParamikoTunnel
            impl = _ParamikoTunnel(self._cfg)
            ok   = impl.start()
        except Exception as exc:
            self.failed.emit(f"Ausnahme beim Verbinden:\n{exc}")
            return
        if ok:
            cfg = self._cfg
            log = (
                f"SSH-Tunnel ge\u00f6ffnet:\n"
                f"  localhost:{cfg['local_port']} \u2192 "
                f"{cfg['remote_host']}:{cfg['remote_port']}"
                f" via {cfg['user']}@{cfg['host']}\n\n"
                "Klicke \"Ollama-Verbindung testen\" um Ollama zu pr\u00fcfen."
            )
            self.connected.emit(impl, "\U0001f7e2  Verbunden", log)
        else:
            self.failed.emit(
                "SSH-Verbindung fehlgeschlagen.\n"
                "Pr\u00fcfe Host, Benutzer und Passwort."
            )


class ClickableImageLabel(QLabel):
    """QLabel subclass that emits `clicked(x, y)` with the click's
    widget-local coordinates on left mouse button press. Used for both
    the calibration frame and the color-calibration board image."""

    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Qt mouse-press handler: emit `clicked` with the click position
        on left-button presses, then defer to the base implementation."""
        if event.button() == Qt.MouseButton.LeftButton:
            pt = event.position().toPoint()
            self.clicked.emit(pt.x(), pt.y())
        super().mousePressEvent(event)


class AdminPanel(QDialog):
    """Modal admin dialog hosting four tabs: Camera device selection,
    board Calibration (homography), Ollama connection settings
    (including a manual-terminal SSH tunnel workflow), and HSV Color
    calibration for the CV detection pipeline.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        frame_provider: Optional[Callable[[], Any]] = None,
        compute_homography: Optional[Callable[[list[list[float]]], Any]] = None,
        camera=None,
        detector=None,
    ) -> None:
        """Initialize dialog state and build all four tabs.

        Args:
            parent: Optional parent widget.
            frame_provider: Optional zero-arg callable returning the
                latest raw camera frame (BGR ndarray, QImage, or
                QPixmap), used for live preview in the Calibration and
                Color tabs.
            compute_homography: Optional fallback callable computing a
                homography from 4 source points, used only if OpenCV's
                own `cv2.findHomography` call fails.
            camera: Optional camera instance (used for device switching
                and calibration reload); None implies demo mode.
            detector: Optional live detector instance, used as a
                fallback frame source if `frame_provider` is unavailable.
        """
        super().__init__(parent)
        self.frame_provider     = frame_provider
        self.compute_homography = compute_homography
        self._camera            = camera
        self._detector          = detector

        self.setWindowTitle("Admin Panel")
        self.setModal(True)
        self.resize(980, 780)

        self.calibration_points: list[QPoint] = []
        self.current_pixmap: Optional[QPixmap] = None
        self.color_defaults = self._build_color_defaults()

        self._ssh_tunnel_impl = None
        self._ssh_worker: Optional[_SshConnectWorker] = None
        self._scan_worker: Optional[_CameraScanWorker] = None

        self._color_tab_stable: Optional[Any] = None

        self._snapshot_board: Optional[np.ndarray] = None
        self._snapshot_matrix: Optional[list[list[str]]] = None
        self._is_snapshot: bool = False

        self._last_board_bgr: Optional[np.ndarray] = None
        self._last_stable_matrix: Optional[list[list[str]]] = None

        self._color_overrides: dict[tuple[int, int], str] = {}
        self._pending_click: Optional[tuple[int, int, QPoint]] = None
        self._color_selected_cell: Optional[tuple[int, int]] = None

        self._color_samples: dict[str, list[tuple[int, int, int]]] = {
            cn: [] for cn in _ALL_COLOR_NAMES
        }

        # Checkboxes for enabling/disabling colors
        self._color_checkboxes: dict[str, QCheckBox] = {}

        self._calib_timer = QTimer(self)
        self._calib_timer.setInterval(100)
        self._calib_timer.timeout.connect(self._refresh_calib_live)

        self._color_timer = QTimer(self)
        self._color_timer.setInterval(250)
        self._color_timer.timeout.connect(self._refresh_color_live)

        self._build_ui()
        self._update_calibration_status()
        self._load_color_ranges_into_table()
        self._load_disabled_state()

        if self.frame_provider is not None:
            self._calib_timer.start()
        else:
            self._refresh_frame()

    def _build_ui(self) -> None:
        """Build the tab widget (Camera, Calibration, Ollama, Color) and
        the bottom status bar."""
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_camera_tab(),      "Kamera")
        self.tabs.addTab(self._build_calibration_tab(), "Kalibrierung")
        self.tabs.addTab(self._build_ollama_tab(),      "Ollama")
        self.tabs.addTab(self._build_color_tab(),       "Farb-Kalibrierung")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs)
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Bereit")

    def _on_tab_changed(self, index: int) -> None:
        """Start/stop the calibration- and color-preview timers based on
        which tab is now active, to avoid unnecessary background work."""
        self._calib_timer.stop()
        self._color_timer.stop()
        if index == 1 and self.frame_provider is not None:
            self._calib_timer.start()
        elif index == 3:
            if not self._is_snapshot:
                self._color_timer.start()
            self._refresh_color_view()

    # ------------------------------------------------------------------
    # Kamera tab
    # ------------------------------------------------------------------

    def _build_camera_tab(self) -> QWidget:
        """Build the Camera tab: device scan button, found-device combo
        box, manual index connect field, and a live camera status
        label."""
        tab  = QWidget()
        root = QVBoxLayout(tab)
        root.setSpacing(10)
        info = QLabel(
            "W\u00e4hle das Kamera-Ger\u00e4t aus oder verbinde per Index.\n"
            "Das Live-Bild siehst du im Tab \u201eKalibrierung\u201c."
        )
        info.setWordWrap(True)
        root.addWidget(info)
        scan_row = QHBoxLayout()
        self._scan_btn = QPushButton("\U0001f50d  Kameras suchen")
        self._scan_btn.setFixedWidth(180)
        self._scan_btn.clicked.connect(self._scan_cameras)
        scan_row.addWidget(self._scan_btn)
        self._scan_status_lbl = QLabel("Noch nicht gesucht")
        self._scan_status_lbl.setStyleSheet("color:#888580;")
        scan_row.addWidget(self._scan_status_lbl)
        scan_row.addStretch(1)
        root.addLayout(scan_row)
        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Gefundene Ger\u00e4te:"))
        self._cam_combo = QComboBox()
        self._cam_combo.setMinimumWidth(260)
        self._cam_combo.setEnabled(False)
        self._cam_combo.addItem("(zuerst \u201eKameras suchen\u201c klicken)")
        select_row.addWidget(self._cam_combo)
        self._cam_apply_btn = QPushButton("\u00dcbernehmen")
        self._cam_apply_btn.setEnabled(False)
        self._cam_apply_btn.clicked.connect(self._apply_camera_selection)
        select_row.addWidget(self._cam_apply_btn)
        select_row.addStretch(1)
        root.addLayout(select_row)
        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("Index manuell:"))
        self._manual_index_spin = QSpinBox()
        self._manual_index_spin.setRange(0, 19)
        self._manual_index_spin.setValue(0)
        self._manual_index_spin.setFixedWidth(70)
        manual_row.addWidget(self._manual_index_spin)
        manual_apply = QPushButton("Verbinden")
        manual_apply.setFixedWidth(100)
        manual_apply.clicked.connect(self._apply_manual_index)
        manual_row.addWidget(manual_apply)
        manual_row.addStretch(1)
        root.addLayout(manual_row)
        self._cam_status_lbl = QLabel()
        self._update_camera_status_label()
        root.addWidget(self._cam_status_lbl)
        root.addStretch(1)
        return tab

    def _scan_cameras(self) -> None:
        """Kick off the background camera scan and update the status
        label while it runs."""
        self._scan_btn.setEnabled(False)
        self._scan_status_lbl.setText("Suche l\u00e4uft \u2026")
        self._scan_status_lbl.setStyleSheet("color:#f0a500;")
        self._scan_worker = _CameraScanWorker(self)
        self._scan_worker.scan_done.connect(self._on_scan_done)
        self._scan_worker.start()

    def _on_scan_done(self, found: list) -> None:
        """Populate the device combo box with the scan results (or show
        a "not found" state) once the background scan finishes.

        Args:
            found: List of discovered camera device indices.
        """
        self._scan_btn.setEnabled(True)
        self._cam_combo.clear()
        if found:
            for idx in found:
                self._cam_combo.addItem(f"Kamera {idx}  (Index {idx})", userData=idx)
            self._cam_combo.setEnabled(True)
            self._cam_apply_btn.setEnabled(True)
            self._scan_status_lbl.setText(f"{len(found)} Ger\u00e4t(e) gefunden: {found}")
            self._scan_status_lbl.setStyleSheet("color:#44d17a;")
        else:
            self._cam_combo.addItem("Keine Kamera gefunden")
            self._scan_status_lbl.setText("Keine Kameras \u2014 Index manuell eingeben")
            self._scan_status_lbl.setStyleSheet("color:#ff6b6b;")
        self.status_bar.showMessage("Kamera-Scan abgeschlossen")

    def _apply_camera_selection(self) -> None:
        """Switch to the camera device currently selected in the combo
        box, if any."""
        idx = self._cam_combo.currentData()
        if idx is not None:
            self._switch_to_device(int(idx))

    def _apply_manual_index(self) -> None:
        """Switch to the camera device index entered manually in the
        spin box."""
        self._switch_to_device(self._manual_index_spin.value())

    def _switch_to_device(self, idx: int) -> None:
        """Attempt to switch the active camera to the given device
        index and report success/failure to the user.

        Args:
            idx: The cv2.VideoCapture device index to switch to.
        """
        if self._camera is None:
            QMessageBox.information(self, "Kamera", "Im Demo-Modus ist kein Kamera-Wechsel m\u00f6glich.")
            return
        ok = self._camera.switch_device(idx)
        self._update_camera_status_label()
        msg = f"Kamera {idx} verbunden." if ok else f"Ger\u00e4t {idx} konnte nicht ge\u00f6ffnet werden."
        (QMessageBox.information if ok else QMessageBox.warning)(self, "Kamera", msg)

    def _update_camera_status_label(self) -> None:
        """Refresh the camera-tab status label to reflect demo mode,
        connected, or disconnected state."""
        if not hasattr(self, "_cam_status_lbl"):
            return
        if self._camera is None:
            self._cam_status_lbl.setText("Modus: Demo")
            self._cam_status_lbl.setStyleSheet("color:#888580;")
        elif hasattr(self._camera, "is_open") and self._camera.is_open():
            self._cam_status_lbl.setText("\U0001f7e2  Kamera verbunden")
            self._cam_status_lbl.setStyleSheet("color:#44d17a; font-weight:600;")
        else:
            self._cam_status_lbl.setText("\U0001f534  Keine Kamera")
            self._cam_status_lbl.setStyleSheet("color:#ff6b6b; font-weight:600;")

    # ------------------------------------------------------------------
    # Kalibrierung tab
    # ------------------------------------------------------------------

    def _build_calibration_tab(self) -> QWidget:
        """Build the Calibration tab: instructional banner, live-frame
        click target, point/reset/load/save buttons, and a warp preview
        panel shown once 4 points are set."""
        tab  = QWidget()
        root = QVBoxLayout(tab)
        root.setSpacing(8)
        banner = QLabel(
            "<b>Live-Kamerabild</b> \u2014 klicke die 4 Ecken deines 5\u00d75-Grids:<br>"
            "&nbsp;&nbsp;1\ufe0f\u20e3 oben-links &nbsp;\u2192&nbsp; "
            "2\ufe0f\u20e3 oben-rechts &nbsp;\u2192&nbsp; "
            "3\ufe0f\u20e3 unten-rechts &nbsp;\u2192&nbsp; "
            "4\ufe0f\u20e3 unten-links<br>"
            "<span style='color:#00dcff;'>Cyan</span> = Raster-Linien &nbsp;| "
            "<span style='color:#00ffb4;'>Gr\u00fcn</span> = Farbmess-Bereich pro Zelle"
        )
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "background:#1a1917; border:1px solid #2e2d2b; border-radius:6px; padding:8px;"
        )
        root.addWidget(banner)
        status_row = QHBoxLayout()
        self.calibration_status_label = QLabel()
        status_row.addWidget(self.calibration_status_label)
        status_row.addStretch(1)
        self._points_label = QLabel("Ecken: 0 / 4")
        self._points_label.setStyleSheet("color:#888580; font-size:12px;")
        status_row.addWidget(self._points_label)
        root.addLayout(status_row)
        self.frame_label = ClickableImageLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setMinimumSize(700, 460)
        self.frame_label.setStyleSheet(
            "background:#111; border:1px solid #2e2d2b; border-radius:6px;"
        )
        self.frame_label.clicked.connect(self._on_frame_clicked)
        root.addWidget(self.frame_label, 1)
        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        for label, slot in [
            ("\u2715 Punkte zur\u00fccksetzen", self._reset_points),
            ("\U0001f4c2 Bestehende laden",  self._load_existing_calibration),
        ]:
            btn = QPushButton(label)
            btn.setFixedWidth(180)
            btn.clicked.connect(slot)
            button_row.addWidget(btn)
        self.compute_button = QPushButton("\u2705 Kalibrierung speichern")
        self.compute_button.setEnabled(False)
        self.compute_button.setFixedWidth(210)
        self.compute_button.setStyleSheet(
            "background:#1e6b3a; color:#c8f0d8; border:1px solid #2e9b5a; "
            "border-radius:8px; padding:6px 12px; font-weight:700;"
        )
        self.compute_button.clicked.connect(self._compute_and_save_calibration)
        button_row.addWidget(self.compute_button)
        button_row.addStretch(1)
        root.addLayout(button_row)
        self._warp_preview_label = QLabel()
        self._warp_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._warp_preview_label.setFixedHeight(0)
        self._warp_preview_label.setStyleSheet(
            "background:#111; border:1px solid #2e2d2b; border-radius:6px;"
        )
        root.addWidget(self._warp_preview_label)
        return tab

    def _refresh_calib_live(self) -> None:
        """QTimer callback: pull the latest live frame and redraw the
        calibration view with any existing corner points overlaid."""
        raw = self._get_frame()
        pm  = self._frame_to_pixmap(raw)
        if pm is None or pm.isNull():
            return
        self.current_pixmap = pm
        self._redraw_frame_with_points()

    def _refresh_frame(self) -> None:
        """One-shot (non-timer) equivalent of `_refresh_calib_live`, used
        when no live frame provider is available."""
        self.current_pixmap = self._frame_to_pixmap(self._get_frame())
        self._redraw_frame_with_points()

    # ------------------------------------------------------------------
    # Ollama tab
    # ------------------------------------------------------------------

    def _build_ollama_tab(self) -> QWidget:
        """Build the Ollama tab: SSH-tunnel connect/disconnect controls,
        connection parameter form (base URL, model, temperature, max
        tokens), and a test-connection button with a results text
        area."""
        tab  = QWidget()
        root = QVBoxLayout(tab)
        root.setSpacing(10)
        ssh_box  = QGroupBox("SSH-Tunnel (Ollama-Server)")
        ssh_form = QFormLayout(ssh_box)
        self._ssh_host_edit = QLineEdit("100.106.237.123")
        ssh_form.addRow("Host", self._ssh_host_edit)
        self._ssh_port_edit = QLineEdit("22")
        self._ssh_port_edit.setFixedWidth(60)
        ssh_form.addRow("Port", self._ssh_port_edit)
        self._ssh_user_edit = QLineEdit("forschungsseminar")
        ssh_form.addRow("Benutzer", self._ssh_user_edit)
        self._ssh_local_port_edit = QLineEdit("11434")
        self._ssh_local_port_edit.setFixedWidth(70)
        ssh_form.addRow("Lokaler Port", self._ssh_local_port_edit)
        root.addWidget(ssh_box)
        ssh_btn_row = QHBoxLayout()
        self._ssh_connect_btn = QPushButton("\U0001f517  SSH-Tunnel \u00f6ffnen")
        self._ssh_connect_btn.setFixedWidth(200)
        self._ssh_connect_btn.clicked.connect(self._on_ssh_connect)
        ssh_btn_row.addWidget(self._ssh_connect_btn)
        self._ssh_disconnect_btn = QPushButton("\u274c  Tunnel schlie\u00dfen")
        self._ssh_disconnect_btn.setFixedWidth(160)
        self._ssh_disconnect_btn.setEnabled(False)
        self._ssh_disconnect_btn.clicked.connect(self._on_ssh_disconnect)
        ssh_btn_row.addWidget(self._ssh_disconnect_btn)
        self._ssh_status_lbl = QLabel("\u26ab  Nicht verbunden")
        self._ssh_status_lbl.setStyleSheet("color:#888580; padding-left:8px;")
        ssh_btn_row.addWidget(self._ssh_status_lbl)
        ssh_btn_row.addStretch(1)
        root.addLayout(ssh_btn_row)
        form_box = QGroupBox("Verbindung und Parameter")
        form     = QFormLayout(form_box)
        self.ollama_url_edit   = QLineEdit(str(getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")))
        form.addRow("Base URL", self.ollama_url_edit)
        self.ollama_model_edit = QLineEdit(str(getattr(config, "OLLAMA_MODEL", "llama3")))
        form.addRow("Model", self.ollama_model_edit)
        temp_wrap   = QWidget()
        temp_layout = QHBoxLayout(temp_wrap)
        temp_layout.setContentsMargins(0, 0, 0, 0)
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(0, 100)
        self.temperature_slider.setValue(30)
        self.temperature_value_label = QLabel("0.30")
        self.temperature_slider.valueChanged.connect(
            lambda v: self.temperature_value_label.setText(f"{v / 100:.2f}")
        )
        temp_layout.addWidget(self.temperature_slider)
        temp_layout.addWidget(self.temperature_value_label)
        form.addRow("Temperature", temp_wrap)
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(32, 8192)
        self.max_tokens_spin.setValue(512)
        form.addRow("Max Tokens", self.max_tokens_spin)
        root.addWidget(form_box)
        self.test_connection_btn = QPushButton("Ollama-Verbindung testen")
        self.test_connection_btn.clicked.connect(self._test_ollama_connection)
        root.addWidget(self.test_connection_btn)
        self.ollama_result = QTextEdit()
        self.ollama_result.setReadOnly(True)
        self.ollama_result.setMinimumHeight(160)
        root.addWidget(self.ollama_result, 1)
        return tab

    def _on_ssh_connect(self) -> None:
        """Open a new terminal window running the configured `ssh -N -L
        ...` port-forward command (platform-specific: cmd on Windows,
        AppleScript/Terminal on macOS, common terminal emulators on
        Linux), then start polling for Ollama reachability through the
        forwarded local port.
        """
        host = self._ssh_host_edit.text().strip()
        user = self._ssh_user_edit.text().strip()
        port = self._ssh_port_edit.text().strip() or "22"
        local_port = self._ssh_local_port_edit.text().strip() or "11434"
        if not host or not user:
            QMessageBox.warning(self, "SSH-Tunnel", "Bitte Host und Benutzer angeben.")
            return

        ssh_cmd = f"ssh -N -L {local_port}:localhost:11434 -p {port} {user}@{host}"

        try:
            system = platform.system()
            if system == "Windows":
                subprocess.Popen(
                    f'start "SSH-Tunnel" cmd /k {ssh_cmd}',
                    shell=True,
                )
            elif system == "Darwin":
                script = f'tell application "Terminal" to do script "{ssh_cmd}"'
                subprocess.Popen(["osascript", "-e", script])
            else:
                opened = False
                for term in ("gnome-terminal", "xterm", "konsole"):
                    try:
                        subprocess.Popen([term, "--", "bash", "-c", ssh_cmd])
                        opened = True
                        break
                    except FileNotFoundError:
                        continue
                if not opened:
                    QMessageBox.warning(
                        self, "SSH-Tunnel",
                        "Kein bekanntes Terminal gefunden. Bitte manuell ausführen:\n" + ssh_cmd,
                    )
                    return
        except Exception as exc:
            QMessageBox.critical(self, "SSH-Tunnel", f"Konnte Terminal nicht öffnen:\n{exc}")
            return

        self._ssh_status_lbl.setText("\U0001f7e1 Terminal geöffnet \u2014 Passwort dort eingeben \u2026")
        self._ssh_status_lbl.setStyleSheet("color:#f0a500; padding-left:8px;")
        self.ollama_result.setPlainText(
            "Im neuen Terminal-Fenster ausgeführt:\n" + ssh_cmd +
            "\n\nBitte dort das Passwort eingeben. Sobald der Tunnel steht, "
            "wird die Verbindung hier automatisch erkannt."
        )
        self._ssh_connect_btn.setEnabled(False)
        self._ssh_disconnect_btn.setEnabled(True)
        self._start_ssh_poll(local_port)

    def _start_ssh_poll(self, local_port: str) -> None:
        """Start (or restart) a 2-second-interval QTimer polling the
        given local port for Ollama reachability.

        Args:
            local_port: The locally forwarded port to poll.
        """
        if not hasattr(self, "_ssh_poll_timer"):
            self._ssh_poll_timer = QTimer(self)
            self._ssh_poll_timer.timeout.connect(self._poll_ssh_tunnel)
        self._ssh_poll_port = local_port
        self._ssh_poll_timer.setInterval(2000)
        self._ssh_poll_timer.start()

    def _poll_ssh_tunnel(self) -> None:
        """QTimer callback: check whether Ollama is now reachable through
        the forwarded local port, and if so, stop polling and update the
        status label/status bar accordingly."""
        url = f"http://localhost:{self._ssh_poll_port}"
        try:
            resp = httpx.get(f"{url}/api/tags", timeout=2)
            ok = resp.status_code == 200
        except Exception:
            ok = False
        if ok:
            self._ssh_poll_timer.stop()
            self._ssh_status_lbl.setText("\U0001f7e2 Tunnel aktiv \u2014 Ollama erreichbar")
            self._ssh_status_lbl.setStyleSheet("color:#44d17a; padding-left:8px; font-weight:600;")
            self.status_bar.showMessage("SSH-Tunnel aktiv, Ollama erreichbar")

    def _on_ssh_disconnect(self) -> None:
        """Handle the "close tunnel" button: stop polling, reset the
        status label/buttons, and remind the user to close the terminal
        window manually (since the tunnel process is not managed
        directly by this app)."""
        if hasattr(self, "_ssh_poll_timer"):
            self._ssh_poll_timer.stop()
        self._ssh_status_lbl.setText("\u26ab Nicht verbunden")
        self._ssh_status_lbl.setStyleSheet("color:#888580; padding-left:8px;")
        self._ssh_disconnect_btn.setEnabled(False)
        self._ssh_connect_btn.setEnabled(True)
        QMessageBox.information(
            self, "SSH-Tunnel",
            "Bitte schließe das SSH-Terminal-Fenster manuell (Strg+C oder Fenster schließen).",
        )

    def closeEvent(self, event) -> None:
        """Qt close-event handler: stop all timers and wait briefly for
        the camera-scan worker thread to finish before closing."""
        self._calib_timer.stop()
        self._color_timer.stop()
        if hasattr(self, "_ssh_poll_timer"):
            self._ssh_poll_timer.stop()
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.wait(2000)
        super().closeEvent(event)

    def _test_ollama_connection(self) -> None:
        """Handle the "test Ollama connection" button: query the
        configured base URL's /api/tags endpoint, report whether the
        configured model was found among the available models, and show
        the result (or error) in the results text area."""
        url = self.ollama_url_edit.text().strip().rstrip("/")
        model = self.ollama_model_edit.text().strip()
        temp = self.temperature_slider.value() / 100.0
        max_tok = self.max_tokens_spin.value()
        if not url:
            self.ollama_result.setPlainText("Base URL fehlt.")
            return
        try:
            resp = httpx.get(f"{url}/api/tags", timeout=4)
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            found = any(n.startswith(model) for n in models)
            nl = "\n- "
            self.ollama_result.setPlainText(
                f"Verbindung OK.\nModel: {model} {'✓' if found else '(nicht gefunden)'}\n"
                f"Temperature: {temp:.2f} | Max Tokens: {max_tok}\n\n"
                f"Modelle:\n- {nl.join(models) if models else 'keine'}"
            )
        except Exception as exc:
            self.ollama_result.setPlainText(f"Fehlgeschlagen:\n{exc}")

    # ------------------------------------------------------------------
    # Farb-Kalibrierung tab
    # ------------------------------------------------------------------

    def _build_color_tab(self) -> QWidget:
        """Build the Color-calibration tab: usage hint banner, active-
        colors checkbox row, snapshot/live controls, the clickable board
        image, sample controls, the HSV-range table, and save/reset
        buttons."""
        tab  = QWidget()
        root = QVBoxLayout(tab)
        root.setSpacing(6)

        hint = QLabel(
            "<b>1.</b> \u201eSnapshot\u201c klicken um das Bild einzufrieren. "
            "<b>2.</b> Kachel anklicken \u2192 richtige Farbe w\u00e4hlen. "
            "<b>3.</b> Mehrere Kacheln anklicken (mind. 2-3 pro Farbe). "
            "<b>4.</b> \u201eHSV berechnen\u201c \u2192 \u201eSpeichern\u201c. "
            "<span style='color:#f0a500;'>Tipp: Nur anklicken wenn Erkennung falsch ist \u2014 "
            "validierte Ranges bleiben als Fallback erhalten.</span>"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "background:#1a1917; border:1px solid #2e2d2b; border-radius:6px; padding:6px;"
        )
        root.addWidget(hint)

        # ── Active colors ──────────────────────────────────────────────
        active_box = QGroupBox("Aktive Farben (deaktivierte Farben werden als 'leer' erkannt)")
        active_box.setStyleSheet(
            "QGroupBox { border:1px solid #2e2d2b; border-radius:6px; margin-top:6px; "
            "color:#888580; font-size:12px; padding:4px; } "
            "QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }"
        )
        active_layout = QHBoxLayout(active_box)
        active_layout.setSpacing(12)
        active_layout.setContentsMargins(8, 12, 8, 6)

        for color_name in _TOGGLEABLE_COLORS:
            cb = QCheckBox()
            cb.setChecked(True)  # default: all active
            cb.setToolTip(f"{color_name} bei der Erkennung ber\u00fccksichtigen")

            # Swatch label
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background:{_COLOR_HEX.get(color_name, '#666')}; "
                "border:1px solid #555; border-radius:2px;"
            )

            name_lbl = QLabel(f"{_COLOR_EMOJI.get(color_name, '')} {color_name}")
            name_lbl.setStyleSheet("color:#f0efed; font-size:12px;")

            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(4)
            cell_layout.addWidget(cb)
            cell_layout.addWidget(swatch)
            cell_layout.addWidget(name_lbl)

            active_layout.addWidget(cell)
            self._color_checkboxes[color_name] = cb

        active_layout.addStretch(1)

        # Apply-button
        apply_active_btn = QPushButton("\u2714 Anwenden & Speichern")
        apply_active_btn.setFixedWidth(190)
        apply_active_btn.setStyleSheet(
            "background:#1e4b6b; color:#c8e8f0; border:1px solid #2e7b9a; "
            "border-radius:6px; padding:4px 10px; font-weight:700;"
        )
        apply_active_btn.clicked.connect(self._apply_active_colors)
        active_layout.addWidget(apply_active_btn)

        root.addWidget(active_box)
        # ────────────────────────────────────────────────────────────────

        ctrl_row = QHBoxLayout()
        self._snapshot_btn = QPushButton("\U0001f4f7  Snapshot")
        self._snapshot_btn.setFixedWidth(130)
        self._snapshot_btn.setStyleSheet(
            "background:#3a2b1a; color:#f0d0a0; border:1px solid #8a5a2a; "
            "border-radius:6px; padding:5px 10px; font-weight:700;"
        )
        self._snapshot_btn.clicked.connect(self._take_snapshot)
        ctrl_row.addWidget(self._snapshot_btn)
        self._live_btn = QPushButton("\u25b6  Live")
        self._live_btn.setFixedWidth(100)
        self._live_btn.setEnabled(False)
        self._live_btn.clicked.connect(self._resume_live)
        ctrl_row.addWidget(self._live_btn)
        self._snap_status_lbl = QLabel("Live")
        self._snap_status_lbl.setStyleSheet("color:#44d17a; padding-left:8px; font-weight:600;")
        ctrl_row.addWidget(self._snap_status_lbl)

        reset_validated_btn = QPushButton("\u21ba  Validierte Werte")
        reset_validated_btn.setFixedWidth(160)
        reset_validated_btn.setToolTip("Setzt alle Ranges auf die validierten Run-4-Werte zur\u00fcck")
        reset_validated_btn.clicked.connect(self._reset_to_validated)
        ctrl_row.addWidget(reset_validated_btn)

        ctrl_row.addStretch(1)
        root.addLayout(ctrl_row)

        self._color_frame_lbl = ClickableImageLabel()
        self._color_frame_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._color_frame_lbl.setMinimumSize(480, 320)
        self._color_frame_lbl.setStyleSheet(
            "background:#0d0f0e; border:1px solid #2e2d2b; border-radius:6px;"
        )
        self._color_frame_lbl.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._color_frame_lbl.clicked.connect(self._on_color_cell_clicked)
        root.addWidget(self._color_frame_lbl, 2)

        sample_row = QHBoxLayout()
        self._sample_count_lbl = QLabel("Samples: 0")
        self._sample_count_lbl.setStyleSheet("color:#888580;")
        sample_row.addWidget(self._sample_count_lbl)
        clear_samples_btn = QPushButton("\u2715 Samples l\u00f6schen")
        clear_samples_btn.setFixedWidth(150)
        clear_samples_btn.clicked.connect(self._clear_samples)
        sample_row.addWidget(clear_samples_btn)
        apply_samples_btn = QPushButton("\U0001f9e0 HSV berechnen")
        apply_samples_btn.setFixedWidth(160)
        apply_samples_btn.setStyleSheet(
            "background:#3a2b6b; color:#d0c8f0; border:1px solid #5a4b9a; "
            "border-radius:6px; padding:5px 10px; font-weight:700;"
        )
        apply_samples_btn.clicked.connect(self._apply_samples_to_table)
        sample_row.addWidget(apply_samples_btn)
        sample_row.addStretch(1)
        root.addLayout(sample_row)

        self._color_cal_table = QTableWidget(0, 8)
        self._color_cal_table.setHorizontalHeaderLabels(
            ["Farbe", "Samples", "H-low", "H-high", "S-low", "S-high", "V-low", "Live HSV"]
        )
        self._color_cal_table.verticalHeader().setVisible(False)
        self._color_cal_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        hdr = self._color_cal_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for c in range(2, 7):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed)
            self._color_cal_table.setColumnWidth(c, 58)
        hdr.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._color_cal_table, 3)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("\u21ba  Zur\u00fccksetzen")
        reset_btn.setFixedWidth(140)
        reset_btn.clicked.connect(self._load_color_ranges_into_table)
        btn_row.addWidget(reset_btn)
        save_btn = QPushButton("\U0001f4be  Speichern")
        save_btn.setFixedWidth(140)
        save_btn.setStyleSheet(
            "background:#1e4b6b; color:#c8e8f0; border:1px solid #2e7b9a; "
            "border-radius:6px; padding:5px 10px; font-weight:700;"
        )
        save_btn.clicked.connect(self._save_color_ranges)
        btn_row.addWidget(save_btn)
        self._color_status_lbl = QLabel()
        self._color_status_lbl.setStyleSheet("color:#888580; padding-left:8px;")
        btn_row.addWidget(self._color_status_lbl)
        btn_row.addStretch(1)
        root.addLayout(btn_row)
        self._populate_color_table()
        return tab

    # ------------------------------------------------------------------
    # Active-colors logic
    # ------------------------------------------------------------------

    def _get_disabled_colors(self) -> list[str]:
        """Return list of color names currently unchecked."""
        return [
            cn for cn, cb in self._color_checkboxes.items()
            if not cb.isChecked()
        ]

    def _apply_active_colors(self) -> None:
        """Remove disabled colors from COLOR_RANGES and save to file."""
        try:
            from vision.cv_live_stream import (
                COLOR_RANGES, _VALIDATED_DEFAULTS,
                _build_ranges_from_dict, save_color_ranges_to_file,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", str(exc))
            return

        disabled = self._get_disabled_colors()

        # Rebuild from validated defaults, skipping disabled colors
        filtered = {
            k: v for k, v in _VALIDATED_DEFAULTS.items()
            if k not in disabled and k != "red2"
        }
        # Also keep red2 only if red is active
        if "red" not in disabled and "red2" in _VALIDATED_DEFAULTS:
            filtered["red2"] = _VALIDATED_DEFAULTS["red2"]

        COLOR_RANGES.clear()
        COLOR_RANGES.update(_build_ranges_from_dict(filtered))

        # Save with _disabled metadata so it survives restart
        save_color_ranges_to_file(COLOR_RANGES)
        self._save_disabled_to_file(disabled)

        # Update table to reflect active set
        self._populate_color_table()

        n_active   = len(_TOGGLEABLE_COLORS) - len(disabled)
        n_disabled = len(disabled)
        msg = (
            f"\u2714 {n_active} Farben aktiv"
            + (f", {n_disabled} deaktiviert: {', '.join(disabled)}" if disabled else "")
        )
        self._color_status_lbl.setText(msg)
        self._color_status_lbl.setStyleSheet(
            "color:#44d17a; padding-left:8px;" if not disabled
            else "color:#f0a500; padding-left:8px;"
        )
        self.status_bar.showMessage(
            f"Aktive Farben gespeichert \u2014 {n_disabled} deaktiviert"
        )

    def _save_disabled_to_file(self, disabled: list[str]) -> None:
        """Persist disabled list as _disabled key in color_ranges.json."""
        try:
            from vision.cv_live_stream import _COLOR_RANGES_FILE
            path = Path(_COLOR_RANGES_FILE)
            data: dict = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            data["_disabled"] = disabled
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[admin_panel] Could not save _disabled: {exc}")

    def _load_disabled_state(self) -> None:
        """Restore checkbox state from color_ranges.json _disabled key."""
        try:
            from vision.cv_live_stream import _COLOR_RANGES_FILE
            path = Path(_COLOR_RANGES_FILE)
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
            disabled = data.get("_disabled", [])
            for cn, cb in self._color_checkboxes.items():
                cb.setChecked(cn not in disabled)
        except Exception:
            pass

    def _reset_to_validated(self) -> None:
        """Restore validated Run-4 defaults into COLOR_RANGES and table."""
        try:
            from vision.cv_live_stream import COLOR_RANGES, _VALIDATED_DEFAULTS, _build_ranges_from_dict, save_color_ranges_to_file
            COLOR_RANGES.clear()
            COLOR_RANGES.update(_build_ranges_from_dict(_VALIDATED_DEFAULTS))
            save_color_ranges_to_file(COLOR_RANGES)
            # Re-enable all checkboxes
            for cb in self._color_checkboxes.values():
                cb.setChecked(True)
            self._save_disabled_to_file([])
            self._populate_color_table()
            self._color_status_lbl.setText("\u21ba Validierte Werte wiederhergestellt")
            self._color_status_lbl.setStyleSheet("color:#44d17a; padding-left:8px;")
            self.status_bar.showMessage("Validierte HSV-Ranges wiederhergestellt und gespeichert")
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", f"{exc}")

    # ------------------------------------------------------------------
    # Snapshot / Live
    # ------------------------------------------------------------------

    def _take_snapshot(self) -> None:
        """Freeze the current live board frame/matrix into
        self._snapshot_board / self._snapshot_matrix, stop the color
        preview timer, and switch the UI into snapshot editing mode."""
        if self._last_board_bgr is None:
            self._refresh_color_live()
        if self._last_board_bgr is None:
            self.status_bar.showMessage("Kein Board-Bild verf\u00fcgbar")
            return
        self._snapshot_board  = self._last_board_bgr.copy()
        self._snapshot_matrix = [list(row) for row in (self._last_stable_matrix or [])]
        self._is_snapshot = True
        self._color_timer.stop()
        self._snap_status_lbl.setText("\U0001f4f7 Snapshot")
        self._snap_status_lbl.setStyleSheet("color:#f0a500; padding-left:8px; font-weight:600;")
        self._snapshot_btn.setEnabled(False)
        self._live_btn.setEnabled(True)
        self.status_bar.showMessage("Snapshot eingefroren \u2014 klicke Kacheln um Farben zuzuweisen")
        self._render_board_image(self._snapshot_board, self._snapshot_matrix or [])

    def _resume_live(self) -> None:
        """Discard the current snapshot and resume the live color
        preview timer."""
        self._is_snapshot = False
        self._snapshot_board  = None
        self._snapshot_matrix = None
        self._snap_status_lbl.setText("Live")
        self._snap_status_lbl.setStyleSheet("color:#44d17a; padding-left:8px; font-weight:600;")
        self._snapshot_btn.setEnabled(True)
        self._live_btn.setEnabled(False)
        self._color_timer.start()
        self.status_bar.showMessage("Live-Modus")

    def _refresh_color_live(self) -> None:
        """QTimer callback for the color-calibration tab: grab a raw
        frame, warp it to the canonical board view, run detection with a
        short-window stabilizer (separate from the main live detector),
        and refresh the board image + live HSV column."""
        from vision.cv_live_stream import GRID_SIZE, get_cell_hsv_median, StableMatrix
        if self._color_tab_stable is None:
            self._color_tab_stable = StableMatrix(window=5, threshold=3)
        raw = self._get_raw_bgr_frame()
        if raw is None:
            self._show_placeholder_in_color_view()
            return
        board = self._warp_to_board(raw)
        if board is None:
            self._show_placeholder_in_color_view(
                "Noch nicht kalibriert \u2014 bitte zuerst den Kalibrierung-Tab nutzen"
            )
            return
        self._last_board_bgr = board
        from vision.cv_live_stream import detect_matrix
        raw_matrix, _ = detect_matrix(board)
        stable_matrix  = self._color_tab_stable.update(raw_matrix)
        self._last_stable_matrix = stable_matrix
        self._render_board_image(board, stable_matrix)
        self._update_live_hsv_column(board, stable_matrix)

    def _refresh_color_view(self) -> None:
        """Redraw the color-calibration board image from whichever
        source is currently active (frozen snapshot or last live
        frame), without pulling a new frame."""
        if self._is_snapshot and self._snapshot_board is not None:
            self._render_board_image(self._snapshot_board, self._snapshot_matrix or [])
        elif self._last_board_bgr is not None:
            self._render_board_image(self._last_board_bgr, self._last_stable_matrix or [])

    def _render_board_image(self, board: np.ndarray, matrix: list[list[str]]) -> None:
        """Render the annotated board image (per-cell color tint overlay,
        cell borders reflecting selection/override state, color-name and
        sample-count labels, and grid lines) and display it in the
        color-calibration image label.

        Args:
            board: The warped board frame (BGR) to annotate.
            matrix: The detected/stabilized color-name matrix for the
                board, used as the base color per cell (overridden by
                any manual `_color_overrides`).
        """
        try:
            from vision.cv_live_stream import GRID_SIZE
        except Exception:
            GRID_SIZE = _GRID_N
        bw, bh = board.shape[1], board.shape[0]
        cs_x   = bw // GRID_SIZE
        cs_y   = bh // GRID_SIZE
        _TINT = {
            "blue":       (200,  80,  30),
            "darkgray":   ( 70,  70,  70),
            "darkgreen":  (  0,  80,   0),
            "lightgray":  (160, 160, 160),
            "lightgreen": ( 40, 180,  40),
            "red":        ( 30,  30, 180),
            "white":      (230, 230, 230),
        }
        annotated = board.copy()
        overlay   = board.copy()
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x1, y1 = c * cs_x, r * cs_y
                x2, y2 = x1 + cs_x, y1 + cs_y
                try:
                    base_color = matrix[r][c]
                except (IndexError, TypeError):
                    base_color = "empty"
                color_name = self._color_overrides.get((r, c), base_color)
                if color_name != "empty":
                    tint = _TINT.get(color_name, (100, 100, 100))
                    cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), tint, cv2.FILLED)
                if self._color_selected_cell == (r, c):
                    border_color = (255, 255, 255)
                    thickness    = 3
                elif (r, c) in self._color_overrides:
                    border_color = (0, 200, 255)
                    thickness    = 2
                else:
                    border_color = (0, 220, 60) if color_name != "empty" else (0, 60, 220)
                    thickness    = 1
                cv2.rectangle(annotated, (x1 + 1, y1 + 1), (x2 - 2, y2 - 2),
                              border_color, thickness)
                font_scale = max(0.28, cs_x / 280.0)
                cv2.putText(
                    annotated, color_name[:9],
                    (x1 + 4, y1 + int(cs_y * 0.28)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    border_color, 1, cv2.LINE_AA,
                )
                n_samples = len(self._color_samples.get(color_name, []))
                if n_samples > 0:
                    cv2.putText(
                        annotated, f"#{n_samples}",
                        (x1 + 4, y1 + int(cs_y * 0.55)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                        (255, 220, 80), 1, cv2.LINE_AA,
                    )
        cv2.addWeighted(overlay, 0.30, annotated, 0.70, 0, annotated)
        for i in range(1, GRID_SIZE):
            cv2.line(annotated, (i * cs_x, 0),  (i * cs_x, bh), (40, 40, 40), 1)
            cv2.line(annotated, (0, i * cs_y),  (bw, i * cs_y), (40, 40, 40), 1)
        pm = self._frame_to_pixmap(annotated)
        if pm and not pm.isNull():
            scaled = pm.scaled(
                self._color_frame_lbl.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._color_frame_lbl.setPixmap(scaled)

    def _update_live_hsv_column(self, board: np.ndarray, stable_matrix: list[list[str]]) -> None:
        """Update the "Live HSV" column of the color-range table with
        the median HSV value across all currently-matching cells for
        each color.

        Args:
            board: The warped board frame (BGR) used to sample HSV.
            stable_matrix: The current stabilized color-name matrix,
                used to group cells by their detected color.
        """
        try:
            from vision.cv_live_stream import COLOR_RANGES, GRID_SIZE, get_cell_hsv_median
        except Exception:
            return
        bw   = board.shape[1]
        cs_x = bw // GRID_SIZE
        color_cells: dict[str, list[tuple[int, int]]] = {cn: [] for cn in COLOR_RANGES}
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cn = stable_matrix[r][c] if stable_matrix else "empty"
                if cn in color_cells:
                    color_cells[cn].append((r, c))
        color_names = sorted(COLOR_RANGES.keys())
        for row_idx, color_name in enumerate(color_names):
            cells = color_cells.get(color_name, [])
            item  = self._color_cal_table.item(row_idx, 7)
            if item is None:
                continue
            if not cells:
                item.setText("\u2014")
                continue
            medians = []
            for (r, c) in cells:
                m = get_cell_hsv_median(board, r, c, cell_size=cs_x)
                if m is not None:
                    medians.append(m)
            if medians:
                item.setText(
                    f"{int(np.median([m[0] for m in medians]))},"
                    f"{int(np.median([m[1] for m in medians]))},"
                    f"{int(np.median([m[2] for m in medians]))}"
                )
            else:
                item.setText("\u2014")

    def _on_color_cell_clicked(self, x: int, y: int) -> None:
        """Handle a click on the color-calibration board image: map the
        widget-local click coordinates back to a board pixel position,
        then to a (row, col) grid cell, and open the color-selection
        context menu for that cell.

        Args:
            x: Click x-coordinate in the image label's local space.
            y: Click y-coordinate in the image label's local space.
        """
        board = self._snapshot_board if self._is_snapshot else self._last_board_bgr
        if board is None:
            self.status_bar.showMessage("Noch kein Board-Bild. Erst Snapshot oder Live-Bild abwarten.")
            return
        displayed  = self._color_frame_lbl.pixmap()
        if displayed is None or displayed.isNull():
            return
        label_size = self._color_frame_lbl.size()
        pm_size    = displayed.size()
        offset_x   = (label_size.width()  - pm_size.width())  // 2
        offset_y   = (label_size.height() - pm_size.height()) // 2
        if not (offset_x <= x <= offset_x + pm_size.width() and
                offset_y <= y <= offset_y + pm_size.height()):
            return
        try:
            from vision.cv_live_stream import GRID_SIZE
        except Exception:
            GRID_SIZE = _GRID_N
        bw, bh = board.shape[1], board.shape[0]
        bx = (x - offset_x) * bw / max(1, pm_size.width())
        by = (y - offset_y) * bh / max(1, pm_size.height())
        cs_x = bw // GRID_SIZE
        cs_y = bh // GRID_SIZE
        col  = max(0, min(GRID_SIZE - 1, int(bx // cs_x)))
        row  = max(0, min(GRID_SIZE - 1, int(by // cs_y)))
        self._color_selected_cell = (row, col)
        self._refresh_color_view()
        gpos = self._color_frame_lbl.mapToGlobal(QPoint(x, y))
        self._pending_click = (row, col, gpos)
        QTimer.singleShot(0, self._show_color_menu)

    def _show_color_menu(self) -> None:
        """Show the deferred color-selection context menu for the
        pending clicked cell (deferred via QTimer.singleShot so the
        board redraw from `_on_color_cell_clicked` happens first), and
        apply the chosen color as an override + HSV sample."""
        if self._pending_click is None:
            return
        row, col, gpos = self._pending_click
        self._pending_click = None
        board = self._snapshot_board if self._is_snapshot else self._last_board_bgr
        try:
            mat = self._snapshot_matrix if self._is_snapshot else self._last_stable_matrix
            cur = (self._color_overrides.get((row, col))
                   or (mat[row][col] if mat else "?"))
        except (IndexError, TypeError):
            cur = "?"
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background:#1a1917; color:#f0efed; border:1px solid #2e2d2b; "
            "padding:2px; font-size:13px; } "
            "QMenu::item { padding:6px 20px 6px 8px; } "
            "QMenu::item:selected { background:#2e2d2b; border-radius:4px; } "
            "QMenu::separator { height:1px; background:#2e2d2b; margin:3px 8px; }"
        )
        header = menu.addAction(f"Zelle {row+1},{col+1}  \u2014  erkannt: {cur}")
        header.setEnabled(False)
        menu.addSeparator()
        for cn in _ALL_COLOR_NAMES:
            icon   = _make_swatch_icon(_COLOR_HEX.get(cn, "#666"))
            emoji  = _COLOR_EMOJI.get(cn, "")
            action = QAction(icon, f"{emoji}  {cn}", self)
            action.setData(cn)
            if cn == cur:
                action.setCheckable(True)
                action.setChecked(True)
            menu.addAction(action)
        chosen = menu.exec(gpos)
        if chosen is None or chosen.data() is None:
            self._color_selected_cell = None
            self._refresh_color_view()
            return
        assigned: str = chosen.data()
        self._color_overrides[(row, col)] = assigned
        if board is not None:
            self._record_sample_from_board(board, row, col, assigned)
        self._refresh_color_view()
        self.status_bar.showMessage(
            f"Zelle ({row+1},{col+1}) \u2192 {assigned}  \u2014 {len(self._color_overrides)} Overrides aktiv"
        )

    def _record_sample_from_board(self, board: np.ndarray, row: int, col: int, color_name: str) -> None:
        """Sample the given cell's median HSV value from `board` and
        append it to `self._color_samples[color_name]`, then refresh the
        sample-count displays.

        Args:
            board: The warped board frame (BGR) to sample from.
            row: Grid row index of the sampled cell.
            col: Grid column index of the sampled cell.
            color_name: The color label the user assigned to this cell.
        """
        try:
            from vision.cv_live_stream import GRID_SIZE, get_cell_hsv_median
        except Exception:
            return
        bw   = board.shape[1]
        cs_x = bw // GRID_SIZE
        hsv  = get_cell_hsv_median(board, row, col, cell_size=cs_x)
        if hsv is None:
            return
        if color_name not in self._color_samples:
            self._color_samples[color_name] = []
        self._color_samples[color_name].append(hsv)
        total = sum(len(v) for v in self._color_samples.values())
        self._sample_count_lbl.setText(f"Samples: {total}")
        self._update_sample_count_in_table()

    def _update_sample_count_in_table(self) -> None:
        """Refresh the "Samples" column of the color-range table with
        the current per-color sample counts."""
        try:
            from vision.cv_live_stream import COLOR_RANGES
        except Exception:
            return
        color_names = sorted(COLOR_RANGES.keys())
        for row_idx, cn in enumerate(color_names):
            item = self._color_cal_table.item(row_idx, 1)
            if item is None:
                continue
            n = len(self._color_samples.get(cn, []))
            item.setText(str(n) if n > 0 else "\u2014")
            item.setForeground(QColor("#44d17a") if n > 0 else QColor("#888580"))

    def _apply_samples_to_table(self) -> None:
        """Compute new HSV low/high bounds per color from the collected
        samples (min/max +/- margins) and write them into the editable
        table columns (does not persist to disk; the user must still
        click Save).
        """
        try:
            from vision.cv_live_stream import COLOR_RANGES, _VALIDATED_DEFAULTS
        except Exception:
            return
        color_names = sorted(COLOR_RANGES.keys())
        updated = 0
        for row_idx, cn in enumerate(color_names):
            samples = self._color_samples.get(cn, [])
            if not samples:
                continue
            hs = [s[0] for s in samples]
            ss = [s[1] for s in samples]
            vs = [s[2] for s in samples]
            h_low  = max(0,   min(hs) - _HSV_H_MARGIN)
            h_high = min(179, max(hs) + _HSV_H_MARGIN)
            s_low  = max(0,   min(ss) - _HSV_S_MARGIN)
            v_low  = max(0,   min(vs) - _HSV_V_MARGIN)
            default_pair = _VALIDATED_DEFAULTS.get(cn)
            if default_pair:
                s_high = int(default_pair[1][1])
                v_high = int(default_pair[1][2])
            else:
                s_high = 255
                v_high = 255
            for col_idx, val in enumerate([h_low, h_high, s_low, s_high, v_low], start=2):
                item = self._color_cal_table.item(row_idx, col_idx)
                if item:
                    item.setText(str(val))
            updated += 1
        if updated:
            self._color_status_lbl.setText(f"\u2713 {updated} Farben aktualisiert")
            self._color_status_lbl.setStyleSheet("color:#f0a500; padding-left:8px;")
            self.status_bar.showMessage(f"{updated} Farb-Ranges berechnet \u2014 bitte speichern!")
        else:
            self.status_bar.showMessage("Keine Samples vorhanden \u2014 zuerst Kacheln anklicken")

    def _clear_samples(self) -> None:
        """Clear all collected color samples and manual cell overrides,
        then refresh the table and board view."""
        for cn in self._color_samples:
            self._color_samples[cn] = []
        self._color_overrides.clear()
        self._color_selected_cell = None
        self._sample_count_lbl.setText("Samples: 0")
        self._update_sample_count_in_table()
        self._refresh_color_view()
        self.status_bar.showMessage("Alle Samples und Overrides gel\u00f6scht")

    def _get_board_size(self) -> tuple[int, int]:
        """Return the (width, height) in pixels the board should be
        warped to, from config, with sensible fallbacks."""
        return int(getattr(config, "BOARD_WIDTH_PX", 800)), int(getattr(config, "BOARD_HEIGHT_PX", 800))

    def _warp_to_board(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """Warp a raw BGR frame onto the canonical board rectangle using
        the saved calibration homography.

        Args:
            bgr: Raw BGR camera frame.

        Returns:
            The warped board image, or None if no calibration file
            exists or the warp fails.
        """
        calib_path = Path(getattr(config, "CALIBRATION_FILE", "calibration.json"))
        if not calib_path.exists():
            return None
        try:
            data = json.loads(calib_path.read_text(encoding="utf-8"))
            H    = np.array(data["homography_matrix"], dtype=np.float32)
            bw, bh = self._get_board_size()
            return cv2.warpPerspective(bgr, H, (bw, bh))
        except Exception:
            return None

    def _show_placeholder_in_color_view(self, msg: str = "Kein Signal") -> None:
        """Render a plain placeholder image with a centered status
        message into the color-calibration board image label (used when
        no frame or no calibration is available).

        Args:
            msg: The status text to display.
        """
        img = QImage(480, 360, QImage.Format.Format_RGB32)
        img.fill(QColor("#0d0f0e"))
        p = QPainter(img)
        p.setPen(QColor("#888580"))
        p.setFont(_safe_font(10))
        p.drawText(img.rect(), Qt.AlignmentFlag.AlignCenter, msg)
        p.end()
        self._color_frame_lbl.setPixmap(QPixmap.fromImage(img))

    def _get_raw_bgr_frame(self) -> Optional[np.ndarray]:
        """Obtain the latest raw BGR frame from whichever source is
        available: the configured frame provider (normalizing QImage /
        QPixmap results to a BGR ndarray), or the live detector's last
        captured frame as a fallback.

        Returns:
            A BGR ndarray copy of the latest frame, or None if no source
            is available.
        """
        if callable(self.frame_provider):
            try:
                raw = self.frame_provider()
                if isinstance(raw, np.ndarray) and raw.size > 0:
                    return raw.copy()
                if isinstance(raw, (QImage, QPixmap)):
                    if isinstance(raw, QPixmap):
                        raw = raw.toImage()
                    raw = raw.convertToFormat(QImage.Format.Format_RGB888)
                    w, h = raw.width(), raw.height()
                    ptr  = raw.bits()
                    ptr.setsize(h * w * 3)
                    arr  = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3)).copy()
                    return arr[:, :, ::-1].copy()
            except Exception:
                pass
        if self._detector is not None and self._detector.last_frame is not None:
            return self._detector.last_frame.copy()
        return None

    def _populate_color_table(self) -> None:
        """Rebuild the color-range table's rows from the current
        COLOR_RANGES, graying out any currently disabled colors."""
        try:
            from vision.cv_live_stream import COLOR_RANGES
        except Exception:
            COLOR_RANGES = {}
        color_names = sorted(COLOR_RANGES.keys())
        self._color_cal_table.setRowCount(len(color_names))
        for row_idx, color_name in enumerate(color_names):
            pairs = COLOR_RANGES.get(color_name, [])
            if pairs:
                lower, upper = pairs[0]
                h_low  = int(lower[0])
                s_low  = int(lower[1])
                v_low  = int(lower[2])
                h_high = int(upper[0])
                s_high = int(upper[1])
            else:
                h_low, h_high, s_low, s_high, v_low = 0, 180, 0, 255, 0
            name_item = QTableWidgetItem(color_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            # Gray out disabled colors in the table
            disabled = self._get_disabled_colors()
            if color_name in disabled:
                name_item.setForeground(QColor("#555553"))
            self._color_cal_table.setItem(row_idx, 0, name_item)
            sample_item = QTableWidgetItem("\u2014")
            sample_item.setFlags(sample_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            sample_item.setForeground(QColor("#888580"))
            self._color_cal_table.setItem(row_idx, 1, sample_item)
            for col_idx, val in enumerate([h_low, h_high, s_low, s_high, v_low], start=2):
                self._color_cal_table.setItem(row_idx, col_idx, QTableWidgetItem(str(val)))
            median_item = QTableWidgetItem("\u2014")
            median_item.setFlags(median_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            median_item.setForeground(QColor("#888580"))
            self._color_cal_table.setItem(row_idx, 7, median_item)

    def _load_color_ranges_into_table(self) -> None:
        """Reset the color-range table to reflect the current
        COLOR_RANGES (discarding any unsaved edits)."""
        self._populate_color_table()
        self.status_bar.showMessage("Farbwerte zur\u00fcckgesetzt")

    def _save_color_ranges(self) -> None:
        """Read the (possibly edited) HSV bounds from the table, rebuild
        COLOR_RANGES from them (preserving each color's existing V-high
        bound and any secondary "red2" range), persist to
        color_ranges.json, and also persist the current disabled-colors
        state."""
        try:
            from vision.cv_live_stream import COLOR_RANGES, save_color_ranges_to_file
            color_names = sorted(COLOR_RANGES.keys())
            new_ranges: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
            for row_idx, color_name in enumerate(color_names):
                h_low  = int(self._color_cal_table.item(row_idx, 2).text())
                h_high = int(self._color_cal_table.item(row_idx, 3).text())
                s_low  = int(self._color_cal_table.item(row_idx, 4).text())
                s_high = int(self._color_cal_table.item(row_idx, 5).text())
                v_low  = int(self._color_cal_table.item(row_idx, 6).text())
                existing = COLOR_RANGES.get(color_name, [])
                v_high = int(existing[0][1][2]) if existing else 255
                lower = np.array([h_low,  s_low,  v_low],  dtype=np.uint8)
                upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
                ranges_for_color = [(lower, upper)]
                if len(existing) > 1:
                    ranges_for_color.append(existing[1])
                new_ranges[color_name] = ranges_for_color
            COLOR_RANGES.clear()
            COLOR_RANGES.update(new_ranges)
            save_color_ranges_to_file(COLOR_RANGES)
            # Also persist disabled state
            self._save_disabled_to_file(self._get_disabled_colors())
            self._color_status_lbl.setText("\u2713 Gespeichert")
            self._color_status_lbl.setStyleSheet("color:#44d17a; padding-left:8px;")
            self.status_bar.showMessage("color_ranges.json gespeichert")
            QMessageBox.information(self, "Farb-Kalibrierung",
                "HSV-Werte \u00fcbernommen und in color_ranges.json gespeichert.")
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Speichern:\n{exc}")

    # ------------------------------------------------------------------
    # Legacy / frame helpers
    # ------------------------------------------------------------------

    def _build_color_defaults(self) -> dict[str, dict]:
        """Build a fallback per-LCZ-class color-range table from
        config.LCZ_CLASSES's HSV midpoints (used only as legacy support;
        the primary color pipeline uses COLOR_RANGES in
        vision.cv_live_stream instead).

        Returns:
            A dict mapping LCZ ID to a dict of display name, hex color,
            and derived HSV bounds.
        """
        defaults: dict[str, dict] = {}
        for lcz_id in LCZ_ORDER:
            entry = config.LCZ_CLASSES.get(lcz_id)
            if entry is None:
                continue
            def _mid(val, fallback: int = 90) -> int:
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    return int((val[0] + val[1]) / 2)
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return fallback
            h = _mid(entry.get("color_hsv_h", 90), fallback=90)
            s = _mid(entry.get("color_hsv_s", 128), fallback=128)
            v = _mid(entry.get("color_hsv_v", 128), fallback=128)
            defaults[lcz_id] = {
                "name":      entry["name"],
                "color_hex": entry["color_hex"],
                "h_min":     max(0,   h - 10),
                "h_max":     min(179, h + 10),
                "s_min":     max(0,   s - 40),
                "v_min":     max(0,   v - 40),
            }
        return defaults

    def _get_frame(self) -> Any:
        """Return the latest frame from the configured frame provider,
        falling back to a synthetic demo frame if no provider is
        configured or it raises/returns None."""
        if callable(self.frame_provider):
            try:
                frame = self.frame_provider()
                if frame is not None:
                    return frame
            except Exception:
                pass
        return self._generate_demo_frame()

    def _generate_demo_frame(self) -> QImage:
        """Render a static placeholder calibration frame (dashed board
        outline, 5x5 grid lines, and corner labels) for use when no real
        camera/frame provider is available."""
        w, h  = 800, 600
        image = QImage(w, h, QImage.Format.Format_RGB32)
        image.fill(QColor("#101010"))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#555"), 2, Qt.PenStyle.DashLine))
        painter.setBrush(QColor("#1c1c1c"))
        board_rect = image.rect().adjusted(80, 60, -80, -60)
        painter.drawRect(board_rect)
        painter.setPen(QPen(QColor("#888"), 1))
        cw = board_rect.width()  / 5
        ch = board_rect.height() / 5
        for c in range(1, 5):
            x = int(board_rect.left() + c * cw)
            painter.drawLine(x, board_rect.top(), x, board_rect.bottom())
        for r in range(1, 5):
            y = int(board_rect.top() + r * ch)
            painter.drawLine(board_rect.left(), y, board_rect.right(), y)
        painter.setPen(QColor("#666"))
        painter.setFont(_safe_font(9))
        for cx, cy, lbl in [
            (board_rect.left() + 4,    board_rect.top()    + 14, "1: oben-links"),
            (board_rect.right() - 94,  board_rect.top()    + 14, "2: oben-rechts"),
            (board_rect.right() - 100, board_rect.bottom() - 4,  "3: unten-rechts"),
            (board_rect.left() + 4,    board_rect.bottom() - 4,  "4: unten-links"),
        ]:
            painter.drawText(int(cx), int(cy), lbl)
        painter.end()
        return image

    def _frame_to_pixmap(self, frame: Any) -> Optional[QPixmap]:
        """Convert a frame of various possible types (QPixmap, QImage,
        grayscale or BGR numpy ndarray) into a QPixmap, falling back to
        the synthetic demo frame if conversion is not possible.

        Args:
            frame: The frame to convert.

        Returns:
            A QPixmap representation of the frame.
        """
        if isinstance(frame, QPixmap):
            return frame
        if isinstance(frame, QImage):
            return QPixmap.fromImage(frame)
        try:
            if isinstance(frame, np.ndarray):
                if frame.ndim == 2:
                    h, w = frame.shape
                    qi   = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_Grayscale8)
                    return QPixmap.fromImage(qi.copy())
                if frame.ndim == 3 and frame.shape[2] == 3:
                    h, w, _ = frame.shape
                    rgb = frame[:, :, ::-1].copy()
                    qi  = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
                    return QPixmap.fromImage(qi.copy())
        except Exception:
            pass
        return QPixmap.fromImage(self._generate_demo_frame())

    def _redraw_frame_with_points(self) -> None:
        """Redraw the calibration tab's live frame with the grid overlay
        (if 4 corners are set), the dashed polygon connecting clicked
        points, numbered corner markers, and corner-order labels; then
        update the points counter, enable/disable the save button, and
        refresh the warp preview."""
        if self.current_pixmap is None or not hasattr(self, "frame_label"):
            return
        img  = self.current_pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        iw, ih = img.width(), img.height()
        ptr  = img.bits()
        ptr.setsize(ih * iw * 3)
        arr  = np.frombuffer(ptr, dtype=np.uint8).reshape((ih, iw, 3)).copy()
        bgr  = arr[:, :, ::-1].copy()
        pts  = self.calibration_points
        if len(pts) == 4:
            _draw_grid_overlay(bgr, pts)
        rgb2   = bgr[:, :, ::-1].copy()
        qi2    = QImage(rgb2.data, iw, ih, rgb2.strides[0], QImage.Format.Format_RGB888)
        canvas = QPixmap.fromImage(qi2.copy())
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if len(pts) >= 2:
            painter.setPen(QPen(QColor("#00ccff"), 2, Qt.PenStyle.DashLine))
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            if len(pts) == 4:
                painter.drawLine(pts[3], pts[0])
        for idx, pt in enumerate(pts):
            painter.setPen(QPen(QColor("#ff3b30"), 2))
            painter.setBrush(QColor("#ff3b30"))
            painter.drawEllipse(pt, 9, 9)
            painter.setPen(QColor("#ffffff"))
            painter.setFont(_safe_font(9))
            painter.drawText(pt + QPoint(-4, 5), str(idx + 1))
            painter.setPen(QColor("#ffdd88"))
            painter.setFont(_safe_font(8))
            painter.drawText(pt + QPoint(13, -9), _CORNER_LABELS[idx])
        painter.end()
        scaled = canvas.scaled(
            self.frame_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.frame_label.setPixmap(scaled)
        self._points_label.setText(f"Ecken: {len(pts)} / 4")
        self.compute_button.setEnabled(len(pts) == 4)
        if len(pts) == 4:
            self._show_warp_preview()
        else:
            self._warp_preview_label.setFixedHeight(0)

    def _show_warp_preview(self) -> None:
        """Compute a homography from the 4 clicked corners to the
        canonical board rectangle, warp the current frame through it,
        overlay grid lines and sampling boxes, and display the result
        in the warp-preview label. Silently collapses the preview panel
        on any failure."""
        try:
            from config import BOARD_WIDTH_PX, BOARD_HEIGHT_PX
            img  = self.current_pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
            w, h = img.width(), img.height()
            ptr  = img.bits()
            ptr.setsize(h * w * 3)
            arr  = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3)).copy()
            bgr  = arr[:, :, ::-1].copy()
            pts  = self.calibration_points
            src  = np.array([[p.x(), p.y()] for p in pts], dtype=np.float32)
            dst  = np.array(
                [[0, 0], [BOARD_WIDTH_PX, 0],
                 [BOARD_WIDTH_PX, BOARD_HEIGHT_PX], [0, BOARD_HEIGHT_PX]],
                dtype=np.float32,
            )
            H, _ = cv2.findHomography(src, dst)
            warped = cv2.warpPerspective(bgr, H, (BOARD_WIDTH_PX, BOARD_HEIGHT_PX))
            step_x = BOARD_WIDTH_PX  // _GRID_N
            step_y = BOARD_HEIGHT_PX // _GRID_N
            for i in range(1, _GRID_N):
                cv2.line(warped, (i * step_x, 0), (i * step_x, BOARD_HEIGHT_PX), (0, 220, 255), 1)
                cv2.line(warped, (0, i * step_y), (BOARD_WIDTH_PX, i * step_y), (0, 220, 255), 1)
            margin_x = int(step_x * _SAMPLE_FRAC / 2)
            margin_y = int(step_y * _SAMPLE_FRAC / 2)
            for row in range(_GRID_N):
                for col in range(_GRID_N):
                    x1 = col * step_x + margin_x
                    y1 = row * step_y + margin_y
                    x2 = (col + 1) * step_x - margin_x
                    y2 = (row + 1) * step_y - margin_y
                    cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 255, 180), 1)
            rgb2  = warped[:, :, ::-1].copy()
            h2, w2, _ = rgb2.shape
            qi = QImage(rgb2.data, w2, h2, rgb2.strides[0], QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qi.copy())
            self._warp_preview_label.setFixedHeight(180)
            self._warp_preview_label.setPixmap(
                pm.scaled(
                    self._warp_preview_label.width(), 180,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        except Exception:
            self._warp_preview_label.setFixedHeight(0)

    def resizeEvent(self, event) -> None:
        """Qt resize-event handler: redraw the calibration frame so the
        overlay/points scale correctly with the new dialog size."""
        super().resizeEvent(event)
        self._redraw_frame_with_points()

    def _on_frame_clicked(self, x: int, y: int) -> None:
        """Handle a click on the calibration frame: map the widget-local
        click position back to image pixel coordinates, then either
        append it as the next corner point (if fewer than 4 are set) or
        replace whichever existing point is closest to the click
        (allowing corner adjustment after all 4 are placed).

        Args:
            x: Click x-coordinate in the image label's local space.
            y: Click y-coordinate in the image label's local space.
        """
        if self.current_pixmap is None or self.frame_label.pixmap() is None:
            return
        displayed  = self.frame_label.pixmap()
        label_size = self.frame_label.size()
        pm_size    = displayed.size()
        offset_x   = (label_size.width()  - pm_size.width())  // 2
        offset_y   = (label_size.height() - pm_size.height()) // 2
        if not (offset_x <= x <= offset_x + pm_size.width() and
                offset_y <= y <= offset_y + pm_size.height()):
            return
        px = (x - offset_x) * self.current_pixmap.width()  / max(1, pm_size.width())
        py = (y - offset_y) * self.current_pixmap.height() / max(1, pm_size.height())
        clicked = QPoint(int(px), int(py))
        if len(self.calibration_points) < 4:
            self.calibration_points.append(clicked)
        else:
            dists = [(p.x() - clicked.x()) ** 2 + (p.y() - clicked.y()) ** 2
                     for p in self.calibration_points]
            self.calibration_points[dists.index(min(dists))] = clicked
        n = len(self.calibration_points)
        self.status_bar.showMessage(
            f"Ecke {min(n, 4)} gesetzt" if n < 4 else
            "Alle 4 Ecken gesetzt \u2014 Vorschau wird berechnet"
        )
        self._redraw_frame_with_points()

    def _reset_points(self) -> None:
        """Clear all clicked calibration corner points and collapse the
        warp preview."""
        self.calibration_points.clear()
        self._warp_preview_label.setFixedHeight(0)
        if self.current_pixmap:
            self._redraw_frame_with_points()
        self.status_bar.showMessage("Punkte zur\u00fcckgesetzt")

    def _load_existing_calibration(self) -> None:
        """Load previously saved corner points from calibration.json (if
        present) back into the UI for review/adjustment."""
        path = Path(getattr(config, "CALIBRATION_FILE", "calibration.json"))
        if not path.exists():
            QMessageBox.information(self, "Kalibrierung", "Keine calibration.json gefunden.")
            return
        try:
            data   = json.loads(path.read_text(encoding="utf-8"))
            points = data.get("src_points") or data.get("points") or []
            self.calibration_points = [QPoint(int(p[0]), int(p[1])) for p in points[:4]]
            self._redraw_frame_with_points()
            self._update_calibration_status(True)
            self.status_bar.showMessage("Kalibrierung geladen")
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", f"Konnte nicht laden:\n{exc}")

    def _compute_and_save_calibration(self) -> None:
        """Compute the homography from the 4 clicked corner points to
        the canonical board rectangle (preferring cv2.findHomography,
        falling back to the injected `compute_homography` callable or an
        identity matrix on failure), then persist it plus the source
        points and image size to calibration.json and trigger a
        calibration reload on the camera if supported."""
        if len(self.calibration_points) != 4:
            return
        src_points = [[float(p.x()), float(p.y())] for p in self.calibration_points]
        try:
            src = np.array(src_points, dtype=np.float32)
            from config import BOARD_WIDTH_PX, BOARD_HEIGHT_PX
            dst = np.array(
                [[0, 0], [BOARD_WIDTH_PX, 0],
                 [BOARD_WIDTH_PX, BOARD_HEIGHT_PX], [0, BOARD_HEIGHT_PX]],
                dtype=np.float32,
            )
            H, _ = cv2.findHomography(src, dst)
            homography = H.tolist()
        except Exception as exc:
            if callable(self.compute_homography):
                try:
                    homography = self.compute_homography(src_points)
                except Exception:
                    homography = self._identity_homography()
            else:
                homography = self._identity_homography()
                QMessageBox.warning(self, "Hinweis",
                    f"OpenCV nicht verf\u00fcgbar ({exc}).\nIdentit\u00e4ts-Homographie gespeichert.")
        payload = {
            "src_points":        src_points,
            "homography_matrix": self._normalise_homography(homography),
            "image_size": [
                self.current_pixmap.width()  if self.current_pixmap else 0,
                self.current_pixmap.height() if self.current_pixmap else 0,
            ],
        }
        path = Path(getattr(config, "CALIBRATION_FILE", "calibration.json"))
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._update_calibration_status(True)
        self.status_bar.showMessage(f"Gespeichert: {path}")
        QMessageBox.information(self, "Kalibrierung", "Kalibrierung gespeichert.")
        if self._camera is not None and hasattr(self._camera, "reload_calibration"):
            self._camera.reload_calibration()

    def _update_calibration_status(self, calibrated: bool | None = None) -> None:
        """Refresh the calibration-tab status label, auto-detecting
        calibrated state from the calibration file's existence if not
        explicitly provided.

        Args:
            calibrated: Explicit calibrated state, or None to
                auto-detect from whether the calibration file exists.
        """
        if calibrated is None:
            calibrated = Path(getattr(config, "CALIBRATION_FILE", "calibration.json")).exists()
        if calibrated:
            self.calibration_status_label.setText("Status: Kalibriert \u2713")
            self.calibration_status_label.setStyleSheet("color:#44d17a; font-weight:600;")
        else:
            self.calibration_status_label.setText("Status: Nicht kalibriert \u2717")
            self.calibration_status_label.setStyleSheet("color:#ff6b6b; font-weight:600;")

    def _identity_homography(self) -> list[list[float]]:
        """Return a 3x3 identity matrix (as nested lists), used as a
        last-resort fallback homography."""
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def _normalise_homography(self, homography: Any) -> list[list[float]]:
        """Normalize a homography of unknown shape (dataclass, dict
        wrapping a matrix under a common key, or raw nested sequence)
        into a plain 3x3 list-of-lists of floats, falling back to the
        identity matrix on any failure.

        Args:
            homography: The homography value to normalize.

        Returns:
            A 3x3 nested list of floats.
        """
        if is_dataclass(homography):
            homography = asdict(homography)
        if isinstance(homography, dict):
            for key in ("homography", "matrix", "H"):
                if key in homography:
                    homography = homography[key]
                    break
        try:
            return [[float(v) for v in row] for row in homography]
        except Exception:
            return self._identity_homography()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = AdminPanel()
    dlg.show()
    sys.exit(app.exec())
