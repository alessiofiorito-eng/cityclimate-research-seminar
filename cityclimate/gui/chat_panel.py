# cityclimate/gui/chat_panel.py
"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: gui/chat_panel.py — Chat panel widget for the CityClimate
Board. Provides a scrollable message history of user/assistant chat
bubbles, a text input row, an "analyze current configuration" button,
and an online/offline connection status indicator with a banner shown
while Ollama is unreachable.
"""
from __future__ import annotations

import time
from typing import Optional, List, TYPE_CHECKING

from PyQt6.QtCore    import Qt, pyqtSignal
from PyQt6.QtGui     import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QScrollArea, QFrame,
    QSizePolicy,
)

from config import THEME

if TYPE_CHECKING:
    from climate.assessment import ClimateMetrics


_MAX_HISTORY = 6   # keep last 6 exchanges (12 messages) — shorter = less prompt bloat


# Keywords that signal the user wants a SPECIFIC cell location, not a strategy
_SPECIFIC_TRIGGERS = [
    "wo ", "wo?", "welches feld", "welche zelle", "welche felder", "welche zellen",
    "wohin", "zeig mir", "ich weiß nicht wo", "wo genau", "an welcher stelle",
    "konkret", "zeig", "welche position",
]


def _wants_specific_location(text: str) -> bool:
    """Return True if the user is explicitly asking for a specific cell position."""
    t = text.lower()
    return any(trigger in t for trigger in _SPECIFIC_TRIGGERS)


class MessageBubble(QFrame):
    """Chat bubble — always shows full content, no internal scroll."""

    def __init__(self, text: str = "", is_user: bool = False, parent: Optional[QWidget] = None) -> None:
        """Build a single chat bubble.

        Args:
            text: Initial bubble text (empty for a streaming assistant
                bubble that will be filled via `append_text`).
            is_user: If True, style the bubble as a user message
                (red-tinted background); otherwise style it as an
                assistant message (surface background, green-tinted
                text).
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.is_user = is_user

        bg = "rgba(207, 24, 32, 160)" if is_user else THEME.get("surface2", "#222120")
        fg = THEME.get("text", "#f0efed") if is_user else "#c8f0d8"

        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border: 1px solid {THEME.get('border', '#2e2d2b')};
                border-radius: 14px;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(0)

        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._label.setFont(QFont("Segoe UI", 9))
        self._label.setStyleSheet(f"color: {fg}; background: transparent; border: none;")
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._label)

        self._text = ""
        if text:
            self.set_text(text)

    def append_text(self, token: str) -> None:
        """Append a streamed token to the bubble's text and refresh the
        displayed label."""
        self._text += token
        self._label.setText(self._text)

    append_token = append_text

    def set_text(self, text: str) -> None:
        """Replace the bubble's full text content."""
        self._text = text
        self._label.setText(text)

    def get_text(self) -> str:
        """Return the bubble's current full text content."""
        return self._text


class ChatPanel(QWidget):
    """The chat sidebar widget: message history, input row, analyze
    button, and connection status indicator/offline banner.

    Emits `send_requested(user_text, history)` when the user sends a
    message, and `analyze_requested()` when the "analyze configuration"
    button is clicked. The actual LLM request/context building is
    handled externally (in app.py); this widget only manages the chat
    UI and message history bookkeeping.
    """

    # Emits (user_text: str, history: list) — app.py builds context_str fresh
    send_requested    = pyqtSignal(str, list)
    analyze_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize chat state and build the UI, starting in the
        offline/disconnected visual state."""
        super().__init__(parent)
        self._current_ai_bubble: Optional[MessageBubble] = None
        self._state_provider = None
        self._connected = False
        self._history: List[dict] = []
        self._pending_user_content: str = ""
        self._build_ui()
        self.set_connection_status(False)

    def _build_ui(self) -> None:
        """Construct the panel's layout: header with status indicator,
        offline banner, scrollable message area, analyze button, and
        the text-input row with a send button."""
        self.setObjectName("ChatPanel")
        self.setStyleSheet(f"""
            QWidget#ChatPanel {{
                background: {THEME.get('bg', '#0d0f0e')};
                color: {THEME.get('text', '#f0efed')};
            }}
            QLineEdit {{
                background: {THEME.get('surface', '#1a1917')};
                color: {THEME.get('text', '#f0efed')};
                border: 1px solid {THEME.get('border', '#2e2d2b')};
                border-radius: 10px;
                padding: 10px 12px;
            }}
            QPushButton {{
                background: {THEME.get('surface2', '#222120')};
                color: {THEME.get('text', '#f0efed')};
                border: 1px solid {THEME.get('border', '#2e2d2b')};
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{ border-color: {THEME.get('orange', '#EC6525')}; }}
            QScrollArea {{ border: none; background: {THEME.get('bg', '#0d0f0e')}; }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Header
        header = QHBoxLayout()
        title = QLabel("Climate Assistant")
        title.setStyleSheet(f"color: {THEME.get('text', '#f0efed')}; font-size: 15px; font-weight: 700;")
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(12, 12)
        self.status_dot.setStyleSheet("border-radius: 6px; background: #7a7a7a;")
        self.status_label = QLabel("Ollama offline")
        self.status_label.setStyleSheet(f"color: {THEME.get('muted', '#888580')}; font-size: 12px;")
        status_wrap = QHBoxLayout()
        status_wrap.setSpacing(6)
        status_wrap.addWidget(self.status_dot)
        status_wrap.addWidget(self.status_label)
        status_widget = QWidget()
        status_widget.setLayout(status_wrap)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(status_widget)
        root.addLayout(header)

        # Offline banner
        self.offline_banner = QFrame()
        self.offline_banner.setStyleSheet(
            f"background: rgba(236, 101, 37, 70); border: 1px solid {THEME.get('orange', '#EC6525')};"
            "border-radius: 10px;"
        )
        banner_layout = QHBoxLayout(self.offline_banner)
        banner_layout.setContentsMargins(12, 8, 12, 8)
        banner_label = QLabel(
            "Ollama ist offline. Die UI ist weiterhin nutzbar; "
            "Chat-Antworten sind erst nach Verbindung verf\u00fcgbar."
        )
        banner_label.setWordWrap(True)
        banner_label.setStyleSheet(f"color: {THEME.get('text', '#f0efed')};")
        banner_layout.addWidget(banner_label)
        root.addWidget(self.offline_banner)

        # Scrollable message area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setContentsMargins(4, 4, 4, 4)
        self.messages_layout.setSpacing(10)
        self.messages_layout.addStretch(1)
        self.scroll_area.setWidget(self.messages_widget)
        root.addWidget(self.scroll_area, stretch=1)

        # Analyse button
        self.analyze_button = QPushButton("\U0001f50d  Konfiguration analysieren")
        self.analyze_button.setStyleSheet(
            "background: #1e3a2f; color: #c8f0d8; "
            "border: 1px solid #2e6b4a; border-radius: 10px; "
            "padding: 8px 14px; font-weight: bold;"
        )
        self.analyze_button.clicked.connect(self._on_analyse_clicked)
        root.addWidget(self.analyze_button)

        # Input row
        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Frage zur aktuellen Konfiguration eingeben\u2026")
        self.input_line.returnPressed.connect(self._on_send_clicked)
        controls.addWidget(self.input_line, stretch=1)
        self.send_button = QPushButton("\u27a4")
        self.send_button.setFixedWidth(38)
        self.send_button.setStyleSheet(
            f"background: {THEME.get('red', '#CF1820')}; color: white; "
            "border-radius: 10px; font-size: 14px; font-weight: bold; padding: 6px;"
        )
        self.send_button.clicked.connect(self._on_send_clicked)
        controls.addWidget(self.send_button)
        root.addLayout(controls)

    # ---------------------------------------------------------------- public API

    def set_state_provider(self, provider) -> None:
        """Register a zero-arg callable that returns the current board
        state (grid, metrics, source label), used elsewhere (e.g. by
        app.py) to build LLM context on demand."""
        self._state_provider = provider

    def set_connection_status(self, connected: bool) -> None:
        """Update the status dot/label, offline banner visibility, and
        enabled state of the analyze/input/send controls based on
        whether Ollama is currently reachable.

        Args:
            connected: True if Ollama is reachable, False otherwise.
        """
        self._connected = connected
        if connected:
            self.status_dot.setStyleSheet("border-radius: 6px; background: #32c26b;")
            self.status_label.setText("Ollama verbunden")
            self.status_label.setStyleSheet(f"color: {THEME.get('text', '#f0efed')}; font-size: 12px;")
            self.offline_banner.hide()
        else:
            self.status_dot.setStyleSheet("border-radius: 6px; background: #d95c5c;")
            self.status_label.setText("Ollama offline")
            self.status_label.setStyleSheet(f"color: {THEME.get('muted', '#888580')}; font-size: 12px;")
            self.offline_banner.show()
        self.analyze_button.setEnabled(connected)
        self.input_line.setEnabled(connected)
        self.send_button.setEnabled(connected)

    def _add_bubble_widget(self, bubble: MessageBubble) -> None:
        """Insert a bubble widget just before the layout's trailing
        stretch, and scroll the message area to the bottom."""
        self.messages_layout.insertWidget(max(0, self.messages_layout.count() - 1), bubble)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        """Scroll the message area's vertical scrollbar to its maximum
        (bottom) position."""
        bar = self.scroll_area.verticalScrollBar()
        bar.setValue(bar.maximum())

    def add_user_message(self, text: str) -> None:
        """Append a new user-styled chat bubble with the given text."""
        self._add_bubble_widget(MessageBubble(text, is_user=True))

    def start_ai_message(self) -> MessageBubble:
        """Create and append a new empty assistant bubble, set it as the
        current streaming target, and return it.

        Returns:
            The newly created MessageBubble.
        """
        bubble = MessageBubble("", is_user=False)
        self._current_ai_bubble = bubble
        self._add_bubble_widget(bubble)
        return bubble

    def add_ai_token(self, token: str) -> None:
        """Append a streamed token to the current assistant bubble
        (creating one first if none is active yet), then scroll to
        bottom.

        Args:
            token: The next chunk of streamed assistant text.
        """
        if self._current_ai_bubble is None:
            self.start_ai_message()
        self._current_ai_bubble.append_text(token)
        self._scroll_to_bottom()

    def finish_ai_message(self, full_text: str = "") -> None:
        """Called when the LLM finishes streaming; saves exchange to history."""
        ai_text = full_text or (self._current_ai_bubble.get_text() if self._current_ai_bubble else "")
        if self._pending_user_content and ai_text:
            self._history.append({"role": "user",      "content": self._pending_user_content})
            self._history.append({"role": "assistant", "content": ai_text})
            if len(self._history) > _MAX_HISTORY * 2:
                self._history = self._history[-(_MAX_HISTORY * 2):]
            self._pending_user_content = ""
        self._current_ai_bubble = None
        self._scroll_to_bottom()

    def _on_analyse_clicked(self) -> None:
        """Handle the "analyze configuration" button: emit
        `analyze_requested` and show a placeholder user-side message
        indicating an analysis is running (no user text/history is
        attached, since the LLM context is built by the receiver of the
        signal)."""
        if not self._connected:
            return
        self.analyze_requested.emit()
        self.add_user_message("\U0001f4ca  Aktuelle Konfiguration wird analysiert \u2026")
        # analysis has no user text, history is passed by app.py via analyze_requested signal

    def _on_send_clicked(self) -> None:
        """Handle the send button/Enter key: read and clear the input
        field, display the user's message, remember it as the pending
        exchange, and emit `send_requested` with the raw text plus a
        copy of the current history."""
        text = self.input_line.text().strip()
        if not text or not self._connected:
            return
        self.input_line.clear()
        self.add_user_message(text)
        self._pending_user_content = text
        # Emit raw user text + current history — app.py builds context_str fresh
        self.send_requested.emit(text, list(self._history))
