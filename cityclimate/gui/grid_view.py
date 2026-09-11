"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: gui/grid_view.py — GridView, a 2-D LCZ grid visualisation
widget (PyQt6). Renders the current board's LCZ tiles with colors and
icons, row/column labels, hover tooltips showing per-cell climate
info, and a right-click context menu for manually assigning an LCZ
class to a cell.
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import (
    QAction, QColor, QFont, QPainter, QPen, QPixmap, QIcon,
)
from PyQt6.QtWidgets import QWidget, QToolTip, QMenu

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LCZ_CLASSES, LCZ_TILE_COLORS, GRID_ROWS, GRID_COLS


# All LCZ IDs that can appear on the physical board (right-click menu)
_LCZ_ORDER = ["2", "5", "6", "9", "A", "D", "G"]


_LCZ_ICON: dict[str, str] = {
    "2":  "🏢",
    "5":  "🏘",
    "6":  "🏠",
    "9":  "🛖",
    "A":  "🌲",
    "D":  "🌿",
    "G":  "💧",
}


# Icon text color: dark tiles → white text, light tiles → dark text
_LCZ_ICON_COLOR: dict[str, QColor] = {
    "2":  QColor(255, 255, 255, 210),
    "5":  QColor( 30,  30,  30, 210),
    "6":  QColor( 30,  30,  30, 210),
    "9":  QColor(255, 255, 255, 210),
    "A":  QColor(255, 255, 255, 210),
    "D":  QColor( 30,  30,  30, 210),
    "G":  QColor(255, 255, 255, 210),
}


_LABEL_MARGIN = 22


def _hex_to_qcolor(hex_str: str, alpha: int = 255) -> QColor:
    """Convert a hex color string to a QColor with the given alpha
    (opacity) applied."""
    c = QColor(hex_str)
    c.setAlpha(alpha)
    return c


def _tile_color(lcz_id: str | None) -> QColor:
    """Return the display QColor for a given LCZ ID, falling back to a
    neutral dark gray for None/unknown IDs (empty cells)."""
    if lcz_id and lcz_id in LCZ_TILE_COLORS:
        return _hex_to_qcolor(LCZ_TILE_COLORS[lcz_id])
    return QColor("#2a2a2a")


def _make_swatch_icon(color_hex: str, size: int = 18) -> QIcon:
    """Render a small square color-swatch icon (with a thin border) for
    use in the right-click LCZ selection menu."""
    pm = QPixmap(size, size)
    pm.fill(QColor(color_hex))
    p = QPainter(pm)
    p.setPen(QPen(QColor("#555"), 1))
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return QIcon(pm)


class GridView(QWidget):
    """Renders a GRID_ROWS × GRID_COLS LCZ grid with row/col labels."""

    cell_clicked = pyqtSignal(int, int)
    cell_changed = pyqtSignal(int, int, str)

    GRID_LINE_CLR = QColor(255, 255, 255, 60)
    SEL_COLOR     = QColor(255, 255, 255, 120)
    LABEL_COLOR   = QColor(136, 133, 128, 220)

    _ROW_LABELS = [chr(ord('A') + i) for i in range(26)]
    _COL_LABELS = [str(i + 1) for i in range(26)]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize an empty grid (all cells None) and enable mouse
        tracking for hover tooltips."""
        super().__init__(parent)
        self.setMouseTracking(True)
        self._grid: List[List[Optional[str]]] = [
            [None] * GRID_COLS for _ in range(GRID_ROWS)
        ]
        self._hovered: Optional[tuple[int, int]] = None
        self._selected: Optional[tuple[int, int]] = None

    def update_grid(self, grid: List[List[Optional[str]]]) -> None:
        """Replace the displayed grid data and trigger a repaint.

        Args:
            grid: The new GRID_ROWS x GRID_COLS grid of LCZ IDs
                (or None for empty cells) to display.
        """
        self._grid = grid
        self.update()

    def _grid_origin(self) -> tuple[int, int]:
        """Return the (x, y) pixel offset of the grid's top-left corner,
        leaving room for row/column labels."""
        return _LABEL_MARGIN, _LABEL_MARGIN

    def _grid_area(self) -> tuple[int, int]:
        """Return the (width, height) in pixels available for the grid
        itself, excluding the label margin."""
        return self.width() - _LABEL_MARGIN, self.height() - _LABEL_MARGIN

    def _cell_size(self) -> tuple[float, float]:
        """Return the (width, height) in pixels of a single grid cell,
        derived from the current widget size."""
        gw, gh = self._grid_area()
        return gw / GRID_COLS, gh / GRID_ROWS

    def _cell_rect(self, row: int, col: int) -> QRect:
        """Return the pixel-space QRect for the given grid cell."""
        ox, oy = self._grid_origin()
        cw, ch = self._cell_size()
        return QRect(ox + int(col * cw), oy + int(row * ch), int(cw), int(ch))

    def _pos_to_cell(self, pos: QPoint) -> Optional[tuple[int, int]]:
        """Map a widget-local pixel position to a (row, col) grid cell,
        or None if the position falls outside the grid area."""
        ox, oy = self._grid_origin()
        cw, ch = self._cell_size()
        x, y = pos.x() - ox, pos.y() - oy
        if x < 0 or y < 0:
            return None
        col, row = int(x / cw), int(y / ch)
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            return row, col
        return None

    def paintEvent(self, event) -> None:
        """Qt paint handler: draw each cell's tile color, hover/selection
        highlight, and LCZ icon, then draw the grid lines and the
        row/column labels."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        ox, oy = self._grid_origin()
        cw, ch = self._cell_size()
        icon_font  = QFont("Segoe UI Emoji", max(8, int(min(cw, ch) * 0.38)))
        label_font = QFont("Segoe UI", max(7, int(_LABEL_MARGIN * 0.52)))
        label_font.setBold(True)

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                rect   = self._cell_rect(row, col)
                lcz_id = self._grid[row][col]

                painter.fillRect(rect, _tile_color(lcz_id))

                if self._hovered == (row, col):
                    painter.fillRect(rect, QColor(255, 255, 255, 40))
                if self._selected == (row, col):
                    painter.fillRect(rect, self.SEL_COLOR)
                    painter.setPen(QPen(Qt.GlobalColor.white, 2))
                    painter.drawRect(rect.adjusted(1, 1, -1, -1))

                if lcz_id and lcz_id in _LCZ_ICON:
                    painter.setFont(icon_font)
                    painter.setPen(_LCZ_ICON_COLOR.get(lcz_id, QColor(255, 255, 255, 200)))
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, _LCZ_ICON[lcz_id])

        painter.setPen(QPen(self.GRID_LINE_CLR, 1))
        gw, gh = self._grid_area()
        for col in range(1, GRID_COLS):
            x = ox + int(col * cw)
            painter.drawLine(x, oy, x, oy + gh)
        for row in range(1, GRID_ROWS):
            y = oy + int(row * ch)
            painter.drawLine(ox, y, ox + gw, y)

        painter.setFont(label_font)
        painter.setPen(self.LABEL_COLOR)
        for row in range(GRID_ROWS):
            y = oy + int(row * self._cell_size()[1])
            label_rect = QRect(0, y, ox - 2, int(self._cell_size()[1]))
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self._ROW_LABELS[row],
            )
        for col in range(GRID_COLS):
            x = ox + int(col * self._cell_size()[0])
            label_rect = QRect(x, 0, int(self._cell_size()[0]), oy - 2)
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom,
                self._COL_LABELS[col],
            )

        painter.end()

    def mouseMoveEvent(self, event) -> None:
        """Qt mouse-move handler: update the hovered cell (triggering a
        repaint if it changed) and show a rich-text tooltip with the
        hovered cell's LCZ name, relative temperature, and population
        density (or an "empty" label if unassigned)."""
        cell = self._pos_to_cell(event.pos())
        if cell != self._hovered:
            self._hovered = cell
            self.update()
        if cell:
            row, col = cell
            lcz_id = self._grid[row][col]
            cell_label = f"{self._ROW_LABELS[row]}{self._COL_LABELS[col]}"
            if lcz_id and lcz_id in LCZ_CLASSES:
                info = LCZ_CLASSES[lcz_id]
                tip = (
                    f"<b>{cell_label} — {info['name']}</b><br>"
                    f"\u0394T: <b>{info['rel_temp']:+.2f} \u00b0C</b><br>"
                    f"Pop-Dichte: <b>{info['pop_density']:,} /km\u00b2</b>"
                )
            else:
                tip = f"<b>{cell_label}</b> — leer"
            QToolTip.showText(event.globalPosition().toPoint(), tip, self)
        else:
            QToolTip.hideText()

    def leaveEvent(self, event) -> None:
        """Qt leave-event handler: clear the hovered cell and repaint
        when the mouse leaves the widget."""
        self._hovered = None
        self.update()

    def mousePressEvent(self, event) -> None:
        """Qt mouse-press handler: on left click, select the cell and
        emit `cell_clicked`; on right click, open the LCZ selection
        context menu for that cell."""
        cell = self._pos_to_cell(event.pos())
        if cell is None:
            return
        row, col = cell
        if event.button() == Qt.MouseButton.LeftButton:
            self._selected = cell
            self.update()
            self.cell_clicked.emit(row, col)
        elif event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(row, col, event.globalPosition().toPoint())

    def _show_context_menu(self, row: int, col: int, global_pos: QPoint) -> None:
        """Build and show the right-click context menu listing all
        assignable LCZ classes (plus an "empty" option) for the given
        cell, and apply the user's choice by updating the grid and
        emitting `cell_changed`.

        Args:
            row: Row index of the clicked cell.
            col: Column index of the clicked cell.
            global_pos: Screen position at which to show the menu.
        """
        cell_label = f"{self._ROW_LABELS[row]}{self._COL_LABELS[col]}"
        menu = QMenu(self)
        menu.setTitle(cell_label)
        menu.setStyleSheet(
            "QMenu { background:#1a1917; color:#f0efed; border:1px solid #2e2d2b; padding: 2px; }"
            "QMenu::item { padding: 5px 18px 5px 6px; }"
            "QMenu::item:selected { background:#2e2d2b; }"
        )
        for lcz_id in _LCZ_ORDER:
            info = LCZ_CLASSES.get(lcz_id)
            if info is None:
                continue
            icon   = _make_swatch_icon(LCZ_TILE_COLORS.get(lcz_id, "#666666"))
            label  = f"{_LCZ_ICON.get(lcz_id, '')}  {info['name']}"
            action = QAction(icon, label, self)
            action.setData(lcz_id)
            menu.addAction(action)
        menu.addSeparator()
        empty_action = QAction("✕  Leer", self)
        empty_action.setData("")
        menu.addAction(empty_action)
        chosen = menu.exec(global_pos)
        if chosen is not None:
            new_id = chosen.data()
            self._grid[row][col] = new_id if new_id else None
            self.update()
            self.cell_changed.emit(row, col, new_id)
