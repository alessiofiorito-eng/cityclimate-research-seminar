# llm/ollama_client.py
"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: llm/ollama_client.py — Async Ollama REST client (OllamaClient)
with proper context injection. Loads connection/model settings from
llm_config.yaml (falling back to config.py, then hardcoded defaults),
runs streaming chat requests on a background thread, and emits Qt
signals for received tokens, completed responses, errors, and
connection status.
"""

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Optional

import httpx
import yaml
from PyQt6.QtCore import QObject, pyqtSignal

from llm.prompt_builder import build_system_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_LLM_CONFIG_PATH = Path(__file__).parent / "llm_config.yaml"


def _load_llm_config() -> dict:
    """Load Ollama connection/model settings from llm_config.yaml.

    Falls back to reading equivalent values from config.py if the YAML
    file is missing or fails to parse, and finally to hardcoded defaults
    if config.py cannot be imported either.

    Returns:
        A dict of LLM configuration settings (host, model, streaming,
        timeout, max_tokens, temperature, fallback_message).
    """
    if _LLM_CONFIG_PATH.exists():
        try:
            with open(_LLM_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            logger.info("llm_config.yaml loaded from %s", _LLM_CONFIG_PATH)
            return cfg
        except Exception as exc:
            logger.warning("Failed to parse llm_config.yaml (%s) – using config.py fallback", exc)

    try:
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
        return {
            "ollama_host":      OLLAMA_BASE_URL,
            "model":            OLLAMA_MODEL,
            "streaming":        True,
            "timeout_seconds":  OLLAMA_TIMEOUT,
            "max_tokens":       300,
            "temperature":      0.15,
            "fallback_message": "KI-Assistent nicht verfügbar.",
        }
    except ImportError:
        return {
            "ollama_host":      "http://localhost:11434",
            "model":            "climate-analyst",
            "streaming":        True,
            "timeout_seconds":  30,
            "max_tokens":       300,
            "temperature":      0.15,
            "fallback_message": "KI-Assistent nicht verfügbar.",
        }


_CFG = _load_llm_config()


OLLAMA_BASE_URL   = _CFG.get("ollama_host",      "http://localhost:11434")
OLLAMA_MODEL      = _CFG.get("model",            "climate-analyst")
OLLAMA_STREAMING  = _CFG.get("streaming",        True)
OLLAMA_TIMEOUT    = _CFG.get("timeout_seconds",  30)
OLLAMA_MAX_TOKENS = _CFG.get("max_tokens",       300)
OLLAMA_TEMP       = _CFG.get("temperature",      0.15)
FALLBACK_MSG      = _CFG.get("fallback_message", "KI-Assistent nicht verfügbar.")


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

class OllamaClient(QObject):
    """
    Async Ollama REST client with proper context injection.

    Key design:
    - SYSTEM_PROMPT: static role/format rules (never changes)
    - context_str (current grid state): injected as a SECOND system message
      so it stays separate from the conversation history and always reflects
      the current board state – preventing stale recommendations.
    - history: previous assistant/user turns (without context blocks)
    - user_text: the current user message
    """

    token_received    = pyqtSignal(str)
    response_complete = pyqtSignal(str)
    error_occurred     = pyqtSignal(str)
    connection_status  = pyqtSignal(bool)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """Initialize the client's cancellation event and background
        thread handle (no request is started yet)."""
        super().__init__(parent)
        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def check_connection(self) -> bool:
        """Synchronously check whether Ollama is reachable at
        OLLAMA_BASE_URL, emitting `connection_status` (and
        `error_occurred` with the fallback message if unreachable).

        Returns:
            True if Ollama responded successfully, False otherwise.
        """
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{OLLAMA_BASE_URL}/api/tags")
                reachable = resp.status_code == 200
        except Exception:
            reachable = False
        if not reachable:
            self.error_occurred.emit(FALLBACK_MSG)
        self.connection_status.emit(reachable)
        return reachable

    def send_message(
        self,
        user_text: str,
        context_str: str = "",
        history: Optional[list[dict]] = None,
    ) -> None:
        """Send a chat message to Ollama on a background thread,
        streaming the response back via Qt signals.

        Any previously running request is cancelled and joined before
        starting the new one, so only one request is ever in flight.

        Args:
            context_str: The current grid state (from
                build_full_prompt). Injected as its own system message,
                NOT mixed into the user message. This way the model
                always sees the current state, independent of the
                conversation history.
            history: List of {role, content} dicts without context
                blocks — pure prior user questions and assistant
                answers only.
            user_text: The current user message text.
        """
        if self._thread and self._thread.is_alive():
            self.cancel()
            self._thread.join(timeout=2.0)
        self._cancel_event.clear()
        self._thread = threading.Thread(
            target=self._run_in_thread,
            args=(user_text, context_str, history or []),
            daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        """Signal the currently running request (if any) to stop
        streaming as soon as possible."""
        self._cancel_event.set()

    def _run_in_thread(self, user_text, context_str, history):
        """Thread entry point: create a fresh asyncio event loop and run
        `_stream_response` to completion on it."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self._stream_response(user_text, context_str, history)
            )
        finally:
            loop.close()

    async def _stream_response(self, user_text, context_str, history):
        """Build the full chat message list and stream Ollama's
        response, emitting a token signal per chunk and a final
        response-complete signal once done (or an error signal on
        failure).

        Message order:
          1. Static system prompt (role/format rules).
          2. Current grid state as a second system message — ALWAYS
             current, never taken from the history cache.
          3. Prior conversation turns (pure questions/answers, no
             context blocks).
          4. The current user message.
        """
        # 1) Static system prompt (role and format rules)
        messages = [{"role": "system", "content": build_system_prompt()}]

        # 2) Current grid state as a second system block
        #    → ALWAYS current, never from the history cache
        if context_str:
            messages.append({
                "role": "system",
                "content": (
                    "Aktueller Spielstand (immer aktuell, überschreibt alles vorherige):\n"
                    + context_str
                ),
            })

        # 3) Prior conversation turns (pure questions/answers only, no context)
        messages.extend(history)

        # 4) Current user message
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model":    OLLAMA_MODEL,
            "messages": messages,
            "stream":   OLLAMA_STREAMING,
            "options": {
                "temperature": OLLAMA_TEMP,
                "num_predict": OLLAMA_MAX_TOKENS,
            },
        }

        assembled = ""
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        self.error_occurred.emit(f"Ollama HTTP {resp.status_code}")
                        self.connection_status.emit(False)
                        return
                    async for raw_line in resp.aiter_lines():
                        if self._cancel_event.is_set():
                            break
                        if not raw_line.strip():
                            continue
                        try:
                            data = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue
                        token = (
                            data.get("message", {}).get("content", "")
                            or data.get("response", "")
                        )
                        if token:
                            assembled += token
                            self.token_received.emit(token)
                        if data.get("done", False):
                            break

        except httpx.ConnectError:
            self.error_occurred.emit(f"{FALLBACK_MSG} (ConnectError: {OLLAMA_BASE_URL})")
            self.connection_status.emit(False)
            return
        except httpx.TimeoutException:
            self.error_occurred.emit(f"Ollama-Anfrage nach {OLLAMA_TIMEOUT}s abgebrochen.")
            self.connection_status.emit(False)
            return
        except Exception as exc:
            self.error_occurred.emit(f"Unerwarteter Fehler: {exc}")
            self.connection_status.emit(False)
            return

        self.connection_status.emit(True)
        if assembled:
            self.response_complete.emit(assembled)
