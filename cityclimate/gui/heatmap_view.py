"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: gui/heatmap_view.py — HeatmapOverlay, a stable, cell-exact
temperature heatmap over GridView.

The overlay must mirror GridView's internal geometry exactly:
  - origin (ox, oy) = (_LABEL_MARGIN, _LABEL_MARGIN) — same as GridView
  - cell sizes derived from the remaining area after subtracting the margin

This ensures every coloured rect sits precisely on top of its LCZ tile.
"""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QColor, QPainter, QFont
from PyQt6.QtWidgets import QWidget

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GRID_ROWS, GRID_COLS


# Must stay in sync with GridView._LABEL_MARGIN
_LABEL_MARGIN = 22


try:
    import matplotlib.colormaps as _colormaps
    _CMAP = _colormaps["RdYlGn_r"]
except Exception:
    try:
        import matplotlib.cm as _cm
        _CMAP = _cm.get_cmap("RdYlGn_r")
    except Exception:
        _CMAP = None


def _norm_to_qcolor(norm: float, alpha: int = 155) -> QColor:
    """Map a normalized value in [0, 1] to a QColor using the RdYlGn_r
    colormap (red = hot, green = cold), falling back to a simple manual
    red/green interpolation if matplotlib is unavailable.

    Args:
        norm: Normalized temperature value, clamped to [0, 1].
        alpha: Alpha (opacity) channel value for the resulting color.

    Returns:
        The corresponding QColor.
    """
    v = max(0.0, min(1.0, float(norm)))
    if _CMAP is not None:
        r, g, b, _ = _CMAP(v)
        return QColor(int(r * 255), int(g * 255), int(b * 255), alpha)
    return QColor(int(255 * v), int(255 * (1.0 - v)), 0, alpha)


_LEGEND_STEPS  = 24
_LEGEND_W      = 14
_LEGEND_H      = 110
_LEGEND_MARGIN = 10


class HeatmapOverlay(QWidget):
    """Transparent overlay: cell-exact rects aligned with GridView labels."""

    CELL_ALPHA = 155

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the overlay as transparent-to-mouse and hidden by
        default, with a stable initial temperature scale."""
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        self._temp_grid: Optional[List[List[Optional[float]]]] = None

        # Stable scale: only update when range actually changes meaningfully
        self._t_min: float = -6.0
        self._t_max: float =  3.0
        self._visible: bool = False
        self.hide()

    # ------------------------------------------------------------------

    def update_heatmap(self, temp_grid_2d: List[List[Optional[float]]]) -> None:
        """Update the displayed temperature grid and (if the value range
        changed meaningfully) rescale the color legend, then repaint if
        the overlay is currently visible.

        Args:
            temp_grid_2d: The new GRID_ROWS x GRID_COLS grid of
                per-cell temperatures (°C), or None for empty cells.
        """
        self._temp_grid = temp_grid_2d
        flat = [v for row in temp_grid_2d for v in row if v is not None]
        if flat:
            lo, hi = min(flat), max(flat)
            span = hi - lo
            if abs(lo - self._t_min) > 0.25 or abs(hi - self._t_max) > 0.25:
                self._t_min = lo
                self._t_max = hi if span > 0.25 else lo + 0.25
        if self._visible:
            self.update()

    def set_visible(self, visible: bool) -> None:
        """Show or hide the heatmap overlay, raising it above sibling
        widgets when shown.

        Args:
            visible: True to show and raise the overlay, False to hide it.
        """
        self._visible = visible
        if visible:
            self.show()
            self.raise_()
            self.update()
        else:
            self.hide()

    # ------------------------------------------------------------------

    def _grid_geometry(self) -> tuple[int, int, float, float]:
        """Return (ox, oy, cell_w, cell_h) matching GridView's layout exactly."""
        ox = oy = _LABEL_MARGIN
        grid_w = self.width()  - _LABEL_MARGIN
        grid_h = self.height() - _LABEL_MARGIN
        cw = grid_w / GRID_COLS
        ch = grid_h / GRID_ROWS
        return ox, oy, cw, ch

    def paintEvent(self, event) -> None:  # noqa: N802
        """Qt paint handler: fill each cell's rect with its normalized
        temperature color (skipping cells with no data), then draw the
        color-scale legend. No-op if the overlay is hidden or has no
        data yet."""
        if not self._visible or self._temp_grid is None:
            return

        ox, oy, cw, ch = self._grid_geometry()
        t_range = max(self._t_max - self._t_min, 0.25)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                val = self._temp_grid[row][col] if self._temp_grid else None
                if val is None:
                    continue
                norm  = (val - self._t_min) / t_range
                color = _norm_to_qcolor(norm, self.CELL_ALPHA)
                # Use same integer-pixel arithmetic as GridView._cell_rect()
                x = ox + int(col * cw)
                y = oy + int(row * ch)
                rw = (ox + int((col + 1) * cw)) - x
                rh = (oy + int((row + 1) * ch)) - y
                painter.fillRect(QRect(x, y, rw, rh), color)

        self._draw_legend(painter, self.width(), self.height())
        painter.end()

    def _draw_legend(self, painter: QPainter, w: int, h: int) -> None:
        """Draw the vertical color-scale legend (gradient bar with a
        border, plus max/mid/min temperature labels) in the bottom-right
        corner of the overlay.

        Args:
            painter: The active QPainter to draw with.
            w: Current overlay width in pixels.
            h: Current overlay height in pixels.
        """
        lw, lh = _LEGEND_W, _LEGEND_H
        x0 = w - _LEGEND_MARGIN - lw - 32
        y0 = h - _LEGEND_MARGIN - lh

        step_h = lh / _LEGEND_STEPS
        for i in range(_LEGEND_STEPS):
            norm  = 1.0 - i / (_LEGEND_STEPS - 1)
            color = _norm_to_qcolor(norm, 210)
            painter.fillRect(QRect(x0, int(y0 + i * step_h), lw, int(step_h) + 1), color)

        painter.setPen(QColor(255, 255, 255, 140))
        painter.drawRect(x0, y0, lw, lh)

        font = QFont("monospace", 8)
        painter.setFont(font)
        painter.setPen(QColor(240, 239, 237, 230))
        painter.drawText(x0 + lw + 4, y0 + 10,          f"{self._t_max:+.1f}°")
        painter.drawText(x0 + lw + 4, y0 + lh // 2 + 4, f"{(self._t_min + self._t_max) / 2:+.1f}°")
        painter.drawText(x0 + lw + 4, y0 + lh,          f"{self._t_min:+.1f}°")
