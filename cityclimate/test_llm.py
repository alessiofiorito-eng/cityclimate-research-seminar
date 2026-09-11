# test_llm.py
"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: test_llm.py — Standalone CLI tester for the LLM module.
Runs WITHOUT the dashboard, WITHOUT a camera, WITHOUT PyQt6.

Directory: cityclimate/
Usage:
    python test_llm.py                   # Demo grid + automatic feedback
    python test_llm.py --chat            # Interactive question mode
    python test_llm.py --grid urban      # Different demo city
    python test_llm.py --grid green      # Green city
    python test_llm.py --no-stream       # Get the response all at once (no streaming)
"""
import sys
import os
import argparse

# Make sure cityclimate/ is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Demo grids (no external dependencies required)
# ---------------------------------------------------------------------------

DEMO_GRIDS = {
    "green": [
        ["A", "A", "B", "B", "D", "D", "A", "A", "B", "D"],
        ["A", "6", "6", "B", "D", "G", "A", "6", "B", "D"],
        ["B", "6", "6", "6", "G", "G", "B", "6", "6", "A"],
        ["D", "6", "5", "5", "G", "D", "6", "5", "5", "B"],
        ["D", "D", "5", "5", "D", "D", "D", "5", "D", "D"],
        ["A", "6", "5", "5", "G", "D", "6", "5", "5", "B"],
        ["B", "6", "6", "6", "G", "G", "B", "6", "6", "A"],
        ["A", "6", "6", "B", "D", "G", "A", "6", "B", "D"],
        ["A", "A", "B", "B", "D", "D", "A", "A", "B", "D"],
        ["D", "D", "D", "A", "A", "A", "D", "D", "D", "A"],
    ],
    "urban": [
        ["2", "2", "2", "5", "5", "5", "2", "2", "5", "5"],
        ["2", "2", "2", "5", "5", "5", "2", "2", "5", "5"],
        ["5", "5", "2", "2", "5", "5", "5", "2", "2", "5"],
        ["5", "5", "2", "2", "5", "5", "5", "2", "2", "5"],
        ["2", "2", "5", "5", "2", "2", "2", "5", "5", "2"],
        ["2", "2", "5", "5", "2", "2", "2", "5", "5", "2"],
        ["5", "5", "2", "2", "5", "5", "5", "2", "2", "5"],
        ["5", "5", "2", "2", "5", "5", "5", "2", "2", "5"],
        ["2", "2", "5", "5", "2", "2", "2", "5", "5", "2"],
        ["2", "2", "5", "5", "2", "2", "2", "5", "5", "2"],
    ],
    "mixed": [
        ["A", "A", "6", "6", "2", "2", "6", "6", "A", "A"],
        ["A", "6", "6", "5", "2", "2", "5", "6", "6", "A"],
        ["6", "6", "5", "5", "5", "5", "5", "5", "6", "6"],
        ["6", "5", "5", "G", "G", "G", "G", "5", "5", "6"],
        ["2", "2", "5", "G", "D", "D", "G", "5", "2", "2"],
        ["2", "2", "5", "G", "D", "D", "G", "5", "2", "2"],
        ["6", "5", "5", "G", "G", "G", "G", "5", "5", "6"],
        ["6", "6", "5", "5", "5", "5", "5", "5", "6", "6"],
        ["A", "6", "6", "5", "2", "2", "5", "6", "6", "A"],
        ["A", "A", "6", "6", "2", "2", "6", "6", "A", "A"],
    ],
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def separator():
    """Print a horizontal separator line to visually divide console output."""
    print("\n" + "─" * 60 + "\n")


def check_ollama():
    """Check whether the Ollama server is reachable.

    Performs a simple synchronous HTTP request against the Ollama
    "/api/tags" endpoint (no Qt event loop involved) to determine
    reachability.

    Returns:
        True if Ollama responded successfully, False otherwise.
    """
    from llm.ollama_client import OllamaClient
    # Simple synchronous check, without Qt
    import urllib.request
    import urllib.error
    from config import OLLAMA_BASE_URL
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def stream_response(prompt: str) -> str:
    """Send a prompt to Ollama and stream the response directly to the console.

    Builds a chat request with the system prompt plus the user prompt,
    enables streaming, and prints each received token as it arrives while
    also assembling the full response text.

    Args:
        prompt: The user-facing prompt/question to send to the model.

    Returns:
        The fully assembled response string.
    """
    import json
    import urllib.request
    import urllib.error
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
    from llm.prompt_builder import SYSTEM_PROMPT

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": True,
        "options": {"temperature": 0.4, "num_predict": 300},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    assembled = ""
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        assembled += token
                        print(token, end="", flush=True)
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except urllib.error.URLError:
        print("\n[Ollama nicht erreichbar — stelle sicher dass 'ollama serve' laeuft]")
    except Exception as e:
        print(f"\n[Fehler: {e}]")

    print()
    return assembled


def get_response(prompt: str) -> str:
    """Non-streaming variant — returns the complete response as a string.

    Sends the same chat request as stream_response() but waits for the
    full response body before returning, instead of printing tokens as
    they arrive.

    Args:
        prompt: The user-facing prompt/question to send to the model.

    Returns:
        The complete response text, or an error placeholder string if the
        request failed.
    """
    import json
    import urllib.request
    import urllib.error
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
    from llm.prompt_builder import SYSTEM_PROMPT

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 300},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["message"]["content"].strip()
    except urllib.error.URLError:
        return "[Ollama nicht erreichbar]"
    except Exception as e:
        return f"[Fehler: {e}]"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_demo(grid_name: str = "mixed", use_stream: bool = True):
    """Run a one-shot feedback session on a demo grid.

    Checks Ollama reachability, builds the full prompt for the selected
    demo grid, prints the resulting city-model context, and then either
    streams or fetches the model's feedback.

    Args:
        grid_name: Key into DEMO_GRIDS selecting which demo city to use.
        use_stream: If True, stream tokens to the console as they arrive;
            otherwise wait for and print the full response at once.
    """
    from llm.prompt_builder import build_full_prompt

    grid = DEMO_GRIDS.get(grid_name, DEMO_GRIDS["mixed"])

    print(f"\n🏙️  CityClimate LLM-Tester")
    print(f"    Grid: '{grid_name}'  |  Ollama: ", end="")

    if not check_ollama():
        print("❌ nicht erreichbar")
        print("   → Tipp: 'ollama serve' auf dem Spark starten")
        return
    print("✅ verbunden")

    separator()
    prompt = build_full_prompt(grid)
    print("📋 Stadtmodell-Kontext:\n")
    print(prompt)
    separator()
    print("🔬 Feedback vom Climate Researcher:\n")

    if use_stream:
        stream_response(prompt)
    else:
        print(get_response(prompt))


def run_chat(grid_name: str = "mixed"):
    """Run an interactive chat session where the user can ask their own questions.

    Checks Ollama reachability, gives an initial automatic overview of the
    selected demo grid, then enters a loop reading user input from the
    console and streaming the model's answer for each question until the
    user exits.

    Args:
        grid_name: Key into DEMO_GRIDS selecting which demo city to use.
    """
    from llm.prompt_builder import build_full_prompt

    grid = DEMO_GRIDS.get(grid_name, DEMO_GRIDS["mixed"])

    print("\n🏙️  CityClimate — Interaktiver Chat")
    print("    Tippe deine Frage und druecke Enter.")
    print("    'exit' oder Ctrl+C zum Beenden.\n")

    if not check_ollama():
        print("❌ Ollama nicht erreichbar. Bitte 'ollama serve' starten.")
        return
    print("✅ Ollama verbunden. Los geht's!")
    separator()

    # Initial automatic overview
    print("🔬 Erster Ueberblick zum Stadtmodell:\n")
    stream_response(build_full_prompt(grid))

    # Interactive loop
    while True:
        try:
            separator()
            question = input("❓ Deine Frage: ").strip()
            if not question or question.lower() in ("exit", "quit", "q", "bye"):
                print("👋 Bis zum naechsten Mal!")
                break

            prompt = build_full_prompt(grid, user_question=question)
            print("\n🔬 Antwort:\n")
            stream_response(prompt)

        except KeyboardInterrupt:
            print("\n\n👋 Beendet.")
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments and dispatch to either interactive chat mode or
    the one-shot demo mode."""
    parser = argparse.ArgumentParser(
        description="CityClimate LLM Standalone Tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python test_llm.py                   # Mixed-Grid analysieren
  python test_llm.py --grid green      # Gruene Stadt
  python test_llm.py --grid urban      # Dichte Stadt
  python test_llm.py --chat            # Fragen stellen
  python test_llm.py --chat --grid urban
        """,
    )
    parser.add_argument(
        "--chat", action="store_true",
        help="Interaktiver Chat-Modus"
    )
    parser.add_argument(
        "--grid",
        choices=list(DEMO_GRIDS.keys()),
        default="mixed",
        help="Demo-Grid auswaehlen (default: mixed)"
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Streaming deaktivieren"
    )
    args = parser.parse_args()

    if args.chat:
        run_chat(grid_name=args.grid)
    else:
        run_demo(grid_name=args.grid, use_stream=not args.no_stream)


if __name__ == "__main__":
    main()
