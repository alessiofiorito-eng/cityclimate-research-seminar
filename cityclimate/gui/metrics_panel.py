# gui/metrics_panel.py  — compact two-column KPI layout
"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: gui/metrics_panel.py — MetricsPanel, the sidebar showing key
climate KPIs (Heat Exposure, mean temperature, population, green/built
fractions) as compact metric cards, plus an embedded Matplotlib bar
chart showing the current LCZ zone distribution.
"""

from collections import Counter

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from config import LCZ_CLASSES, LCZ_TILE_COLORS, THEME


_FONT_SIZE_HE    = 26
_FONT_SIZE_SMALL = 18


class MetricCard(QFrame):
    """A single KPI display card with a small title label and a large
    value label, optionally rendered in a bigger font (`big=True`) for
    the headline metric."""

    def __init__(self, title, value="—", accent=None, big=False, parent=None):
        """Build the card's title/value labels and apply the dark theme
        styling.

        Args:
            title: The KPI's display name shown above the value.
            value: Initial displayed value text.
            accent: Optional custom text color for the value label;
                defaults to the theme's text color.
            big: If True, use the larger headline font size (for the
                primary Heat Exposure card).
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._font_size = _FONT_SIZE_HE if big else _FONT_SIZE_SMALL
        self.title_label = QLabel(title)
        self.value_label = QLabel(value)

        title_color  = THEME.get("muted",    "#888580")
        text_color   = THEME.get("text",     "#f0efed")
        border       = THEME.get("border",   "#2e2d2b")
        surface      = THEME.get("surface2", "#222120")
        accent_color = accent or text_color

        self.setObjectName("MetricCard")
        self.setStyleSheet(
            f"""
            QFrame#MetricCard {{
                background: {surface};
                border: 1px solid {border};
                border-radius: 10px;
            }}
            QLabel {{ border: none; background: transparent; }}
            """
        )
        self.title_label.setStyleSheet(
            f"color: {title_color}; font-size: 10px; font-weight: 600; letter-spacing: 0.4px;"
        )
        self.value_label.setStyleSheet(
            f"color: {accent_color}; font-size: {self._font_size}px; font-weight: 700;"
        )
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value, color=None):
        """Update the card's displayed value text and optionally its
        text color.

        Args:
            value: New value text to display.
            color: Optional new text color; defaults to the theme's
                text color if omitted.
        """
        self.value_label.setText(value)
        c = color or THEME.get("text", "#f0efed")
        self.value_label.setStyleSheet(
            f"color: {c}; font-size: {self._font_size}px; font-weight: 700;"
        )


class LCZChartCanvas(FigureCanvas):
    """Matplotlib bar-chart canvas showing the current LCZ zone
    distribution, with bar colors matching each zone's tile color on
    the board."""

    def __init__(self, parent=None):
        """Create the underlying Matplotlib Figure/Axes styled to match
        the dark theme, and configure the canvas's Qt size policy."""
        self.figure = Figure(figsize=(3.8, 2.4), dpi=96)
        self.figure.patch.set_facecolor(THEME.get("surface2", "#222120"))
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(210)
        self.setMaximumHeight(240)

    def update_chart(self, grid) -> None:
        """Recompute and redraw the LCZ-class distribution bar chart
        for the given grid.

        Counts occurrences of each non-empty LCZ ID in the grid, sorts
        them (numeric IDs first, then alphabetic), and draws one bar
        per class colored to match its board tile color. Shows a
        placeholder message instead if the grid has no assigned cells.

        Args:
            grid: The current GRID_ROWS x GRID_COLS grid of LCZ IDs (or
                None for empty cells).
        """
        self.ax.clear()
        self.ax.set_facecolor(THEME.get("surface2", "#222120"))

        flat   = [cell for row in grid for cell in row if cell is not None]
        counts = Counter(flat)

        if not counts:
            self.ax.text(
                0.5, 0.5, "Keine Zonen platziert",
                ha="center", va="center",
                color=THEME.get("muted", "#888580"),
                fontsize=10, transform=self.ax.transAxes,
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            for spine in self.ax.spines.values():
                spine.set_visible(False)
            self.draw_idle()
            return

        def _sort_key(k):
            try: return (0, int(k))
            except ValueError: return (1, k)

        labels     = sorted(counts.keys(), key=_sort_key)
        values     = [counts[lbl] for lbl in labels]
        # Use LCZ_TILE_COLORS — identical to what GridView paints on the tiles
        bar_colors = [LCZ_TILE_COLORS.get(lbl, "#666666") for lbl in labels]
        names      = [LCZ_CLASSES.get(lbl, {}).get("name", lbl) for lbl in labels]
        label_color = THEME.get("muted", "#888580")

        x_pos = range(len(labels))
        bars = self.ax.bar(
            x_pos, values,
            color=bar_colors,
            edgecolor=THEME.get("border", "#2e2d2b"),
            linewidth=0.8,
        )

        self.ax.set_title(
            "Zonenverteilung",
            color=THEME.get("text", "#f0efed"),
            fontsize=9, pad=4,
        )
        self.ax.set_xticks(list(x_pos))
        self.ax.set_xticklabels(
            names, rotation=35, ha="right",
            fontsize=7, color=label_color,
        )
        self.ax.tick_params(axis="y", colors=THEME.get("muted", "#888580"), labelsize=7)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_color(THEME.get("border",   "#2e2d2b"))
        self.ax.spines["bottom"].set_color(THEME.get("border", "#2e2d2b"))
        self.ax.grid(axis="y", color=THEME.get("border", "#2e2d2b"), alpha=0.4, linewidth=0.6)
        self.ax.set_axisbelow(True)

        for bar, val in zip(bars, values):
            self.ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(val),
                ha="center", va="bottom",
                color=THEME.get("text", "#f0efed"),
                fontsize=7,
            )

        self.figure.tight_layout(pad=0.8)
        self.draw_idle()


class MetricsPanel(QWidget):
    """The metrics sidebar: a 2-column grid of MetricCards (Heat
    Exposure spanning both columns, then mean temperature/population
    and green/built fractions) plus the embedded LCZ distribution
    chart."""

    def __init__(self, parent=None):
        """Build the KPI card grid and the chart panel, and apply the
        dark theme styling."""
        super().__init__(parent)
        self.setObjectName("MetricsPanel")
        self.setStyleSheet(
            f"""
            QWidget#MetricsPanel {{
                background: {THEME.get('bg', '#0d0f0e')};
                color: {THEME.get('text', '#f0efed')};
            }}
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        kpi_widget = QWidget()
        kpi_grid   = QGridLayout(kpi_widget)
        kpi_grid.setContentsMargins(0, 0, 0, 0)
        kpi_grid.setSpacing(8)
        kpi_grid.setColumnStretch(0, 1)
        kpi_grid.setColumnStretch(1, 1)

        self.he_card        = MetricCard("WÄRMEBELASTUNG  (Heat Exposure)", "—", big=True)
        self.mean_temp_card = MetricCard("Mittlere Temp.",   "—")
        self.total_pop_card = MetricCard("Bevölkerung",      "—")
        self.green_card     = MetricCard("Grünanteil",       "—")
        self.built_card     = MetricCard("Bebauungsanteil",  "—")

        kpi_grid.addWidget(self.he_card, 0, 0, 1, 2)
        kpi_grid.addWidget(self.mean_temp_card, 1, 0)
        kpi_grid.addWidget(self.total_pop_card, 1, 1)
        kpi_grid.addWidget(self.green_card,  2, 0)
        kpi_grid.addWidget(self.built_card,  2, 1)

        root.addWidget(kpi_widget)

        chart_frame = QFrame()
        chart_frame.setStyleSheet(
            f"background: {THEME.get('surface2', '#222120')};"
            f"border: 1px solid {THEME.get('border', '#2e2d2b')}; border-radius: 10px;"
        )
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(8, 8, 8, 4)
        chart_layout.setSpacing(4)
        self.chart_canvas = LCZChartCanvas()
        chart_layout.addWidget(self.chart_canvas)
        root.addWidget(chart_frame)

        root.addStretch(1)

    def update_metrics(self, metrics, grid) -> None:
        """Refresh all KPI cards and the distribution chart from a new
        ClimateMetrics result.

        Heat Exposure is color-coded as a traffic light: green if
        negative (net cooling), yellow up to +1.5°C, red above that.

        Args:
            metrics: A ClimateMetrics instance (or any object exposing
                the same attributes) for the current grid.
            grid: The current GRID_ROWS x GRID_COLS grid of LCZ IDs,
                passed through to the distribution chart.
        """
        he             = getattr(metrics, "heat_exposure",    0.0)
        mean_temp      = getattr(metrics, "mean_temp",        0.0)
        total_pop      = getattr(metrics, "total_population", 0.0)
        green_fraction = getattr(metrics, "green_fraction",   0.0)
        built_fraction = getattr(metrics, "built_fraction",   0.0)

        if he < 0:
            he_color = "#32c26b"
        elif he <= 1.5:
            he_color = "#e6c65b"
        else:
            he_color = "#e05a5a"

        self.he_card.set_value(f"{he:+.2f} °C", he_color)
        self.mean_temp_card.set_value(f"{mean_temp:+.2f} °C")
        self.total_pop_card.set_value(f"{total_pop:,.0f}")
        self.green_card.set_value(f"{green_fraction * 100:.1f}%")
        self.built_card.set_value(f"{built_fraction * 100:.1f}%")
        self.chart_canvas.update_chart(grid)
