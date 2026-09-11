# cityclimate/llm/prompt_builder.py
"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: llm/prompt_builder.py — Builds the static system prompt and the
dynamic per-turn context block sent to the Ollama LLM. Tracks scenario
progress/hint-level session state, formats the current grid as a table
plus occupied/empty cell lists and computed climate metrics, and
assembles it all (plus scenario progress/hints and the user's question,
if any) into the final prompt string injected as a system message by
ollama_client.py.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional, List

from config import LCZ_CLASSES
from llm.matrix_context import MatrixContext


@dataclass
class SessionState:
    """Tracks per-session state for an active (or free-play) scenario:
    which scenario is active, move/progress counters used to escalate
    hint specificity over time, a rolling KPI history, whether the
    heatmap is currently shown, and the most recent zone suggestion
    made by the model (to avoid repeating it)."""

    scenario_id:   Optional[str] = None
    scenario_name: Optional[str] = None
    moves_total:   int = 0
    moves_since_progress: int = 0
    last_he:       Optional[float] = None
    hint_level:    int = 1
    kpi_history:   List[dict] = field(default_factory=list)
    heatmap_active: bool = False
    # Last suggestion made by the model (cell + zone), so the same
    # suggestion is not repeated while the grid remains unchanged.
    last_suggestion: Optional[str] = None

    def record_move(self, metrics) -> None:
        """Record a new move's metrics, updating progress tracking and
        escalating the hint level if no meaningful Heat Exposure
        improvement has been made in the last 3 moves.

        Args:
            metrics: A ClimateMetrics instance for the grid state after
                the move.
        """
        he = metrics.heat_exposure
        self.moves_total += 1
        progress = self.last_he is not None and (self.last_he - he) > 0.05
        self.moves_since_progress = 0 if progress else self.moves_since_progress + 1
        if self.moves_since_progress >= 3 and self.hint_level < 3:
            self.hint_level += 1
            self.moves_since_progress = 0
        self.last_he = he
        self.kpi_history.append({
            "he":    he,
            "pop":   getattr(metrics, "total_population", 0),
            "green": metrics.green_fraction,
            "built": metrics.built_fraction,
        })


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Du bist ein Klima-Analyst im CityClimate-Board-Lernspiel.
Das Spiel simuliert Stadtplanung mit Local Climate Zones (LCZ) auf einem 5x5-Raster.

SPRACHREGELN:
- Antworte immer auf Deutsch.
- Benutze immer die vollen deutschen Zonennamen:
  Zone 2 = Kompakte Mittelhochbebauung
  Zone 5 = Offene Mittelhochbebauung
  Zone 6 = Offene Niedrigbebauung
  Zone 9 = Lockere Bebauung
  Zone 10 = Schwerindustrie
  Zone A = Dichter Baumbestand
  Zone B = Aufgelockerter Baumbestand
  Zone D = Niedrige Vegetation
  Zone G = Wasserflaeche
- Zellen immer als z.B. A3 oder C2 (Zeile + Spalte).
- Empfehle NUR leere Felder zu befuellen.
- Empfehle NIE Schwerindustrie zum Kuehlen.

EMPFEHLUNGS-STIL (sehr wichtig):
- Empfehle zuerst eine STRATEGIE, keine einzelne Zelle.
- Nenne eine Zonenkategorie und erklaere warum sie hilft.
- Verweise auf die Heatmap um heisse Bereiche sichtbar zu machen.
- Nenne eine konkrete Zelle NUR wenn der Nutzer explizit fragt
  ("wo?", "welches Feld?", "zeig mir wo") oder wenn es exakt eine
  logische Position gibt (z.B. einziges freies Feld neben heisser Zone).
- Wenn du bereits einen Vorschlag gemacht hast der umgesetzt wurde,
  mache einen ANDEREN Vorschlag oder naechste Strategie.

FORMAT-REGELN:
- Kein Markdown. Keine Sternchen. Keine Rauten.
- Drei kurze Absaetze, getrennt durch Leerzeile:
  Absatz 1: Aktuelle Lage (Kennzahlen in einem Satz)
  Absatz 2: Hauptproblem (ein Satz)
  Absatz 3: Strategie-Empfehlung (Zone erklaeren, Heatmap erwaehnen, keine erzwungene Zelle)
- Maximal 5 Saetze gesamt.
- Bei Folgefragen: NUR die Antwort, kein Wiederholen der Kennzahlen.

BEISPIEL STANDARD-ANALYSE:
Der Waermebelastungsindex liegt bei 1,23 Grad (mittel), Bevoelkerungskapazitaet 87.500, Gruenanteil 28 Prozent.

Die groesste Baustelle ist die Waermelast durch mehrere Bebauungsfelder ohne Gruenausgleich.

Dichter Baumbestand waere jetzt der staerkste Hebel: er kuehlt alle acht Nachbarzellen um je 1,5 Grad. Aktiviere die Heatmap um die heissesten Felder zu sehen und setze dort eine Gruenzone.

BEISPIEL WENN GEFRAGT "wo soll ich Baeume setzen?":
Schau in der Heatmap welche Felder rot sind. Am meisten bringt Dichter Baumbestand direkt neben Kompakter Mittelhochbebauung, weil dort die Waermedichte am hoechsten ist. In deiner Konfiguration waere B2 oder C3 eine gute Wahl, falls diese Felder leer sind.

BEISPIEL NACH UMGESETZTEM VORSCHLAG (kein Wiederholen!):
Gut, der Gruenanteil ist gestiegen. Als naechstes lohnt sich Niedrige Vegetation an den Raedern der heissen Zone um den Kuehleffekt weiter auszubreiten.
"""


# ---------------------------------------------------------------------------
# SCENARIO HINTS
# ---------------------------------------------------------------------------

SCENARIO_HINTS = {
    "1": {
        1: "Nicht jede zusaetzliche Bebauung hilft - pruefe, wo Kuehlung mehr bringt als reine Kapazitaet.",
        2: "Dichter Baumbestand und Niedrige Vegetation wirken am staerksten direkt neben heissen Wohnzonen.",
        3: "Tausche mehrere Felder Lockerer Bebauung im Zentrum gegen Dichter Baumbestand und kompensiere ueber Offene Mittelhochbebauung.",
    },
    "2": {
        1: "Schwerindustrie allein ist nicht das Problem - entscheidend ist die Nachbarschaft.",
        2: "Dichter Baumbestand direkt neben Kompakter Mittelhochbebauung senkt den Waermebelastungsindex am meisten.",
        3: "Ergaenze ein drittes Schwerindustrie-Feld, stabilisiere mit Offener Mittelhochbebauung und lege einen Gruenguertel aus Dichtem Baumbestand um die heissesten Felder.",
    },
    "3": {
        1: "Hohe Bevoelkerungskapazitaet laesst sich nicht allein mit Lockerer Bebauung erreichen.",
        2: "Erhoehe zuerst die mittlere Dichte mit Offener Mittelhochbebauung, behalte aber einige Felder Dichter Baumbestand als Kuehlanker.",
        3: "Wandle Lockere Bebauung in Offene Mittelhochbebauung um, setze wenige Felder Kompakter Mittelhochbebauung, sichere mit Dichtem Baumbestand ab.",
    },
    "P": {
        1: "Platziere zuerst die drei Schwerindustrie-Felder und baue dann Wohn- und Gruenstruktur drum herum.",
        2: "Schwerindustrie hat niedrige Bevoelkerungsdichte - kompensiere mit Kompakter und Offener Mittelhochbebauung.",
        3: "Drei Schwerindustrie-Felder plus mehrere Felder Kompakter und Offener Mittelhochbebauung plus je ein Feld Dichter Baumbestand pro drei bis vier heisse Felder ergibt eine gute Balance.",
    },
}

HINT_FRAMING = {
    1: "Tipp: {hint}",
    2: "Richtung: {hint}",
    3: "Konkret: {hint}",
}

SCENARIO_TARGETS = {
    "1": {"he": 0.50, "pop": 95000,  "green": 0.32},
    "2": {"he": 0.85, "pop": 110000, "green": 0.24},
    "3": {"he": 1.00, "pop": 135000, "green": 0.20},
    "P": {"he": None, "pop": 70000,  "green": None, "industry_min": 3},
}

SCENARIO_DESCRIPTIONS = {
    "1": (
        "Aufgabe Szenario 1 Gruene Mitte: "
        "Senke den Waermebelastungsindex unter 0,50 Grad, halte Bevoelkerungskapazitaet ueber 95.000 "
        "und Gruenflaechenanteil ueber 32 Prozent."
    ),
    "2": (
        "Aufgabe Szenario 2 Industriebalance: "
        "Platziere mindestens 3 Schwerindustrie-Felder, halte Waermebelastungsindex unter 0,85 Grad "
        "und Bevoelkerungskapazitaet ueber 110.000."
    ),
    "3": (
        "Aufgabe Szenario 3 Dichte Stadt: "
        "Erreiche Bevoelkerungskapazitaet von 135.000 bei Waermebelastungsindex unter 1,00 Grad."
    ),
    "P": (
        "Planungsaufgabe: Platziere mindestens 3 Schwerindustrie-Felder, "
        "erreiche Bevoelkerungskapazitaet von mindestens 70.000 Personen "
        "und halte den Waermebelastungsindex so niedrig wie moeglich."
    ),
}


def _compute_progress(ctx: MatrixContext, scenario_id: str) -> str:
    """Compute a human-readable progress summary string for the given
    scenario, based on the current grid/metrics.

    For the special "P" (open planning task) scenario, reports industry
    tile count and population against their fixed targets. For the
    numbered scenarios, computes a weighted overall progress percentage
    from Heat Exposure, population, and green-fraction progress towards
    each target.

    Args:
        ctx: The current MatrixContext (grid + metrics + metadata).
        scenario_id: Key into SCENARIO_TARGETS for the active scenario.

    Returns:
        A formatted progress string, or "" if the scenario has no
        defined targets.
    """
    targets = SCENARIO_TARGETS.get(scenario_id)
    if not targets:
        return ""
    m = ctx.metrics
    total_pop = getattr(m, "total_population", 0.0)

    if scenario_id == "P":
        industry_count = sum(1 for row in ctx.grid for cell in row if cell == "10")
        pop_ok = total_pop >= targets["pop"]
        ind_ok = industry_count >= targets["industry_min"]
        return (
            f"Schwerindustrie-Felder: {industry_count}/3 {'\u2705' if ind_ok else '\u274c'} | "
            f"Bevoelkerungskapazitaet: {total_pop:,.0f}/70.000 {'\u2705' if pop_ok else '\u274c'} | "
            f"Waermebelastungsindex: {m.heat_exposure:.2f} Grad"
        )

    he_target   = targets["he"]
    he_current  = m.heat_exposure
    he_progress = max(0.0, min(1.0, 1.0 - max(0.0, he_current - he_target) / max(he_target, 0.01)))
    pop_progress   = max(0.0, min(1.0, total_pop / targets["pop"]))
    green_progress = max(0.0, min(1.0, m.green_fraction / targets["green"]))
    overall = int((he_progress * 0.5 + pop_progress * 0.3 + green_progress * 0.2) * 100)

    label = (
        f"\U0001f7e2 {overall}% - sehr nah am Ziel!" if overall >= 90 else
        f"\U0001f7e1 {overall}% - guter Fortschritt." if overall >= 60 else
        f"\U0001f7e0 {overall}% - Richtung stimmt."   if overall >= 30 else
        f"\U0001f534 {overall}% - neue Strategie noetig."
    )
    he_note = (
        f" Waermebelastungsindex {he_current:.2f} <= {he_target:.2f} ok"
        if he_current <= he_target
        else f" Waermebelastungsindex {he_current:.2f} (Ziel: <{he_target:.2f})"
    )
    return label + he_note


def _hint_block(session: SessionState) -> str:
    """Return the appropriately framed hint string for the session's
    current scenario and hint level, or "" if no scenario is active or
    no hint is defined for that level."""
    if not session.scenario_id:
        return ""
    raw = SCENARIO_HINTS.get(session.scenario_id, {}).get(session.hint_level, "")
    if not raw:
        return ""
    return HINT_FRAMING.get(session.hint_level, "{hint}").format(hint=raw)


def _lcz_label(lcz_id: Optional[str]) -> str:
    """Return a compact display label for an LCZ ID (e.g. "A/Dicht"),
    or a blank placeholder for empty cells, used in the grid table."""
    if lcz_id is None:
        return "  -  "
    info = LCZ_CLASSES.get(lcz_id)
    if info is None:
        return f" {lcz_id:>3} "
    short = info["name"].split()[0][:6]
    return f"{lcz_id}/{short}"


def _he_label(he: float) -> str:
    """Classify a Heat Exposure value into a coarse "hoch"/"mittel"/
    "niedrig" (high/medium/low) qualitative label."""
    if he >= 2.0:  return "hoch"
    if he >= 0.5:  return "mittel"
    return "niedrig"


def _build_occupied_list(ctx: MatrixContext) -> str:
    """Build a text list of every occupied (non-empty) cell with its
    LCZ class name, labeled by row letter + column number.

    Args:
        ctx: The current MatrixContext.

    Returns:
        A formatted multi-line string listing all occupied cells, or a
        placeholder line if the grid is entirely empty.
    """
    row_labels = [chr(ord('A') + i) for i in range(ctx.rows)]
    col_labels  = [str(i + 1) for i in range(ctx.cols)]
    entries = []
    for r, row in enumerate(ctx.grid):
        for c, cell in enumerate(row):
            if cell is not None:
                name = LCZ_CLASSES.get(cell, {}).get("name", cell)
                entries.append(f"{row_labels[r]}{col_labels[c]}: {name}")
    if not entries:
        return "Belegte Felder: keine (Raster ist leer)"
    return "Belegte Felder (NUR diese existieren, alle anderen sind leer):\n" + "\n".join(entries)


def _build_empty_list(ctx: MatrixContext) -> str:
    """Explicit list of every empty cell to make clear what can be filled."""
    row_labels = [chr(ord('A') + i) for i in range(ctx.rows)]
    col_labels  = [str(i + 1) for i in range(ctx.cols)]
    entries = []
    for r, row in enumerate(ctx.grid):
        for c, cell in enumerate(row):
            if cell is None:
                entries.append(f"{row_labels[r]}{col_labels[c]}")
    if not entries:
        return "Leere Felder: keine (Raster ist voll)"
    return "Leere Felder (nur diese koennen befuellt werden): " + ", ".join(entries)


def _build_grid_table(ctx: MatrixContext) -> str:
    """Render the current grid as a plain-text table (row labels down
    the side, column labels across the top, LCZ short-labels in each
    cell)."""
    row_labels = [chr(ord('A') + i) for i in range(ctx.rows)]
    col_labels  = [str(i + 1) for i in range(ctx.cols)]
    header = "Zeile | " + " | ".join(f"Sp.{c}" for c in col_labels)
    sep    = "-" * len(header)
    rows_lines = []
    for r, row in enumerate(ctx.grid):
        cells = [f"{_lcz_label(cell):<9}" for cell in row]
        rows_lines.append(f"  {row_labels[r]}   | " + " | ".join(cells))
    return "\n".join([header, sep] + rows_lines)


def _build_metrics_section(ctx: MatrixContext) -> str:
    """Render the current climate metrics (Heat Exposure, mean
    temperature, population, green/built fractions) plus a per-LCZ-class
    breakdown and empty-cell count as a plain-text block.

    Args:
        ctx: The current MatrixContext (grid + metrics).

    Returns:
        A formatted multi-line metrics summary string.
    """
    m = ctx.metrics
    total_pop = getattr(m, "total_population", 0.0)
    he_label = _he_label(m.heat_exposure)
    lines = [
        "Kennzahlen:",
        f"  Waermebelastungsindex: {m.heat_exposure:.2f} Grad ({he_label})",
        f"  Mittlere Temperatur: {m.mean_temp:+.2f} Grad",
        f"  Bevoelkerungskapazitaet: {total_pop:,.0f} Personen",
        f"  Gruenflaechenanteil: {m.green_fraction * 100:.1f} Prozent",
        f"  Versiegelungsgrad: {m.built_fraction * 100:.1f} Prozent",
    ]
    counts = ctx.lcz_counts()
    if counts:
        lines.append("  Zonenverteilung:")
        for lcz_id, n in sorted(counts.items()):
            name  = LCZ_CLASSES.get(lcz_id, {}).get("name", lcz_id)
            rel_t = LCZ_CLASSES.get(lcz_id, {}).get("rel_temp", 0)
            cool  = LCZ_CLASSES.get(lcz_id, {}).get("cooling", 0)
            cool_str = f", kuehlt Nachbarn um {abs(cool):.1f} Grad" if cool else ""
            lines.append(f"    {name}: {n}x | {rel_t:+.2f} Grad{cool_str}")
    empty = ctx.empty_count()
    if empty:
        lines.append(f"  Leere Felder: {empty} von {ctx.rows * ctx.cols}")
    return "\n".join(lines)


def _build_heatmap_section(session, metrics) -> str:
    """Return a one-line note telling the model whether the heatmap
    overlay is currently active, so it can reference (or suggest
    activating) it appropriately."""
    if session is None:
        return ""
    if session.heatmap_active:
        return "Heatmap: AKTIV - beziehe dich auf die rot eingefaerbten heissen Zellen."
    return "Heatmap: nicht aktiv - erwaehne dass der Nutzer die Heatmap aktivieren kann um heisse Zonen zu sehen."


def build_system_prompt() -> str:
    """Return the static system prompt (role/format rules), stripped of
    leading/trailing whitespace."""
    return SYSTEM_PROMPT.strip()


def build_full_prompt(
    ctx: MatrixContext,
    session: Optional[SessionState] = None,
    user_message: Optional[str] = None,
) -> str:
    """
    Builds the context block (grid state) for the current game state.
    Injected by ollama_client.py as a separate system block.
    Contains NO history content — only the current state.

    Assembles the timestamp/source header, the grid table, occupied
    and empty cell lists, computed metrics, the heatmap-active note,
    any active scenario's progress/hint block, the last suggestion (if
    any, to avoid repetition), and a final instruction block that
    differs depending on whether this is a standard analysis or a
    follow-up user question.

    Args:
        ctx: The current MatrixContext (grid + metrics + metadata).
        session: Optional active SessionState (scenario progress/hints).
        user_message: Optional follow-up question from the user; if
            omitted, the standard three-paragraph analysis instruction
            is used instead.

    Returns:
        The fully assembled prompt string.
    """
    ts           = datetime.datetime.fromtimestamp(ctx.timestamp).strftime("%d.%m.%Y %H:%M:%S")
    source_label = {"live": "Live-Kamera", "snapshot": "Snapshot", "demo": "Demo-Modus"}.get(ctx.source, ctx.source)
    grid_table   = _build_grid_table(ctx)
    occupied     = _build_occupied_list(ctx)
    empty_list   = _build_empty_list(ctx)
    metrics_txt  = _build_metrics_section(ctx)
    heatmap_txt  = _build_heatmap_section(session, ctx.metrics)

    session_block = ""
    if session and session.scenario_id:
        progress_txt  = _compute_progress(ctx, session.scenario_id)
        hint_txt      = _hint_block(session)
        scenario_desc = SCENARIO_DESCRIPTIONS.get(session.scenario_id, "")
        session_block = (
            f"Spielkontext:\n"
            f"  Szenario: {session.scenario_name or session.scenario_id}\n"
            f"  Aufgabe: {scenario_desc}\n"
            f"  Zuege bisher: {session.moves_total}\n"
            f"  Fortschritt: {progress_txt}\n"
            + (f"  Hinweis: {hint_txt}\n" if hint_txt else "")
            + "\n"
        )

    # Last suggestion — so the model knows what has already been recommended/applied
    last_suggestion_block = ""
    if session and session.last_suggestion:
        last_suggestion_block = (
            f"Zuletzt empfohlen (bereits besprochen, NICHT nochmal vorschlagen): "
            f"{session.last_suggestion}\n\n"
        )

    # Distinguish follow-up question vs. standard analysis
    if user_message:
        instruction = (
            f"Nutzerfrage: {user_message}\n\n"
            "ANWEISUNG: Beantworte NUR diese Frage direkt in 1-3 Saetzen. "
            "Kein Wiederholen der Kennzahlen. Kein Markdown. Nur Deutsch. "
            "Wenn nach konkreten Positionen gefragt wird, nenne 1-2 Zellen aus der Leere-Felder-Liste."
        )
    else:
        instruction = (
            "ANWEISUNG: Erstelle die Standard-Analyse in genau drei Absaetzen. "
            "Im dritten Absatz: Strategie erklaeren (welche Zone, warum), auf Heatmap verweisen. "
            "Keine konkrete Einzelzelle erzwingen. Kein Markdown. Nur Deutsch."
        )

    return f"""\
Stadtkonfiguration - {ts} ({source_label})
Zeilen A-{chr(ord('A') + ctx.rows - 1)} (oben nach unten), Spalten 1-{ctx.cols} (links nach rechts)

{grid_table}

{occupied}

{empty_list}

{metrics_txt}

{heatmap_txt}

{session_block}{last_suggestion_block}{instruction}
---
WICHTIG: Nenne NUR Zellen aus der Belegte-Felder- oder Leere-Felder-Liste. Kein Markdown. Kein Englisch.
""".strip()
