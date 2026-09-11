# CityClimate Board — Process Flowcharts

Research Seminar, Summer Term 2026 · TH Köln
Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

This document collects the key process/decision flows of the CityClimate
Board application as flowcharts. For the system's static architecture and the
runtime *sequence* diagrams (live detection cycle, LLM request/response), see
[`architecture.md`](./architecture.md). For the physical setup, see
[`versuchsaufbau.md`](./versuchsaufbau.md).

## Contents

1. [Application Startup](#1-application-startup)
2. [Calibration Workflow](#2-calibration-workflow)
3. [Color Calibration Workflow](#3-color-calibration-workflow)
4. [Chat / LLM Request Decision Flow](#4-chat--llm-request-decision-flow)
5. [Scenario / Planning Task Lifecycle](#5-scenario--planning-task-lifecycle)

---

## 1. Application Startup

Covers `main.py` from process start to the running `MainWindow`, including the
camera-mode decision based on CLI flags (see `_build_camera()` and
`MainWindow.__init__()`).

```mermaid
flowchart TD
    Start([python main.py ...]) --> Parse[Parse CLI arguments]
    Parse --> LogCfg[Configure logging]
    LogCfg --> Mode{Which camera flag?}
    Mode -->|--realsense| RS[Create RealSenseCamera]
    Mode -->|--camera| RC[Create RealCamera with device_index]
    Mode -->|--demo / default| MC[Create MockCamera with initial_grid]
    RS --> RSCheck{Camera opened?}
    RSCheck -->|No| WarnRS[Log warning: fallback mode]
    RSCheck -->|Yes| CamReady
    WarnRS --> CamReady
    RC --> RCCheck{Camera opened?}
    RCCheck -->|No| WarnRC[Log warning: fallback mode]
    RCCheck -->|Yes| CamReady
    WarnRC --> CamReady
    MC --> CamReady[Camera object ready]
    CamReady --> QtInit[Create QApplication]
    QtInit --> MWInit[Create MainWindow]
    MWInit --> BuildUI[Build toolbar and central widgets]
    BuildUI --> DemoCheck{demo mode?}
    DemoCheck -->|Yes| LoadDemoGrid[Load initial demo grid]
    DemoCheck -->|No| SkipDemo[Skip - grid comes from camera/CV]
    LoadDemoGrid --> StartDetector
    SkipDemo --> StartDetector[Start LiveStreamDetector thread]
    StartDetector --> StartTimer[Start 500ms update QTimer]
    StartTimer --> LLMCheck{use_llm?}
    LLMCheck -->|Yes| OllamaInit[Check Ollama connection<br/>and start 5s status-poll timer]
    LLMCheck -->|No| ShowWindow
    OllamaInit --> ShowWindow[Show MainWindow]
    ShowWindow --> Loop([Qt event loop running])
```

**Notes:**
- Failure to open a real camera (RealSense or webcam) does **not** stop the
  app — it logs a warning and continues in a fallback/no-signal state so the
  user can still calibrate once a camera is connected, or fall back to
  `--demo`.
- `demo` in `MainWindow` is derived as `not (args.camera or args.realsense)`,
  i.e. `--demo` is effectively the default whenever no hardware flag is given.

---

## 2. Calibration Workflow

Covers the Admin Panel's Calibration tab (`gui/admin_panel.py`), i.e. how a
raw camera frame becomes a homography-corrected board view.

```mermaid
flowchart TD
    Open([Open Admin Panel - Calibration tab]) --> Live[Show live frame]
    Live --> Click[User clicks a corner]
    Click --> Count{4 points set?}
    Count -->|No| Live
    Count -->|Yes| Preview[Compute preview homography<br/>and draw grid overlay]
    Preview --> Adjust{User happy with corners?}
    Adjust -->|No, re-click near existing point| Replace[Replace nearest existing point]
    Replace --> Preview
    Adjust -->|Reset| ResetPts[Clear all points]
    ResetPts --> Live
    Adjust -->|Yes| Save[Click Save calibration]
    Save --> Compute[compute_homography via cv2.findHomography]
    Compute --> Write[Write src_points and homography_matrix<br/>to calibration.json]
    Write --> Reload[Camera.reload_calibration if available]
    Reload --> Done([Calibration active for all future frames])
```

**Notes:**
- Corner order must be top-left → top-right → bottom-right → bottom-left; this
  order defines which board edge maps to which output edge.
- If `cv2.findHomography` is unavailable/fails, the panel falls back to an
  injected `compute_homography` callable, and finally to an identity matrix
  (with a warning shown to the user) rather than crashing.
- Loading an *existing* `calibration.json` (via "Load existing") re-populates
  the 4 corner points so they can be reviewed/adjusted without starting over.

---

## 3. Color Calibration Workflow

Covers the Admin Panel's Color Calibration tab, used to fine-tune the HSV
ranges the CV pipeline uses to classify tile colors.

```mermaid
flowchart TD
    Open([Open Admin Panel - Color Calibration tab]) --> LiveView[Live board view<br/>with detected colors overlaid]
    LiveView --> Freeze{Take snapshot?}
    Freeze -->|No| LiveView
    Freeze -->|Yes| Frozen[Board frozen - snapshot mode]
    Frozen --> ClickCell[User clicks a cell]
    ClickCell --> Menu[Color selection context menu]
    Menu --> Assign[Assign correct LCZ color to cell]
    Assign --> Sample[Record HSV sample for that color]
    Sample --> More{More cells to label?}
    More -->|Yes| ClickCell
    More -->|No| Compute[Click Compute HSV]
    Compute --> Ranges[Compute new H/S/V bounds<br/>from min/max of samples plus margin]
    Ranges --> Review[Review values in editable table]
    Review --> SaveOk{Satisfied?}
    SaveOk -->|No, adjust manually| Review
    SaveOk -->|Yes| SaveFile[Click Save - writes color_ranges.json]
    SaveFile --> ResumeLive[Resume live / reload color ranges]
    ResumeLive --> Done([Updated ranges active in detection pipeline])
```

**Notes:**
- Colors can also be individually **disabled** (treated as always empty)
  via checkboxes — persisted as the `_disabled` key in `color_ranges.json`.
- A "Reset to validated values" action restores the hand-tuned
  `_VALIDATED_DEFAULTS` baked into `vision/cv_live_stream.py`, discarding any
  custom samples/ranges.
- Saving triggers `MainWindow._reload_color_ranges()`, which also resets the
  live detector's `StableMatrix` buffer so stale frames don't linger.

---

## 4. Chat / LLM Request Decision Flow

Covers what happens between a user action in `ChatPanel` and a rendered
assistant response, focusing on the *branching* logic (LLM enabled/disabled,
manual question vs. automatic analysis).

```mermaid
flowchart TD
    Trigger{User action}
    Trigger -->|Types message and sends| SendMsg[ChatPanel._on_send_clicked]
    Trigger -->|Clicks Analyze configuration| Analyze[ChatPanel._on_analyse_clicked]

    SendMsg --> LLMCheck1{use_llm enabled?}
    Analyze --> LLMCheck2{use_llm enabled?}

    LLMCheck1 -->|No| Disabled1[Show LLM disabled message]
    LLMCheck2 -->|No| Disabled2[Show LLM disabled message]

    LLMCheck1 -->|Yes| BuildCtx1[Build MatrixContext from<br/>current grid and cached metrics]
    LLMCheck2 -->|Yes| BuildCtx2[Build MatrixContext from<br/>current grid and cached metrics]

    BuildCtx1 --> Prompt1[build_full_prompt<br/>with user_message]
    BuildCtx2 --> Prompt2[build_full_prompt<br/>without user_message]

    Prompt1 --> Send[OllamaClient.send_message]
    Prompt2 --> Send

    Send --> Cancel{Previous request<br/>still running?}
    Cancel -->|Yes| Abort[Cancel and join old thread]
    Cancel -->|No| Spawn
    Abort --> Spawn[Spawn new background thread<br/>with fresh asyncio loop]

    Spawn --> Stream[Stream chat request to Ollama]
    Stream --> TokenLoop{Token received?}
    TokenLoop -->|Yes| Emit[Emit token_received<br/>append to chat bubble]
    Emit --> TokenLoop
    TokenLoop -->|done or error| Complete{Success?}
    Complete -->|Yes| FinishOk[Emit response_complete<br/>save to history<br/>scan for zone suggestion]
    Complete -->|No| FinishErr[Emit error_occurred<br/>show error, finish empty bubble]
```

**Notes:**
- The grid/metrics context is always rebuilt fresh at request time — never
  taken from the chat history — so recommendations are always grounded in the
  *current* board state (see `architecture.md`, "LLM Integration").
- `_on_llm_response_complete()` scans the finished response text for known LCZ
  zone keywords (German/English) to remember the model's most recent
  suggestion (`SessionState.last_suggestion`), so it can be referenced (and
  avoided as a repeat) in the next prompt.

---

## 5. Scenario / Planning Task Lifecycle

Covers selecting a guided scenario (or the open planning task), playing
through it, and reaching a result — spanning `gui/app.py`
(`_activate_scenario`, `PlanningTimerWidget`) and `llm/prompt_builder.py`
(`SessionState`, hint escalation).

```mermaid
flowchart TD
    Select([User selects scenario in toolbar dropdown]) --> IsNone{Scenario is Free play?}
    IsNone -->|Yes| FreeState[Reset SessionState<br/>no scenario_id]
    IsNone -->|No| LoadScenario[Load scenario start_grid and targets]
    LoadScenario --> ApplyGrid[Apply start grid to camera/GUI]
    ApplyGrid --> ComputeInit[Compute initial metrics]
    ComputeInit --> IsPlanning{Scenario is the planning task?}
    IsPlanning -->|Yes| StartTimer[Activate PlanningTimerWidget<br/>5 minute countdown]
    IsPlanning -->|No| NoTimer[No countdown - free-form scenario]

    StartTimer --> UserPlays[User edits grid or<br/>asks chat for hints]
    NoTimer --> UserPlays

    UserPlays --> Move[Each cell edit triggers record_move]
    Move --> ProgressCheck{Heat Exposure improved<br/>by more than 0.05?}
    ProgressCheck -->|Yes| ResetStreak[Reset moves_since_progress]
    ProgressCheck -->|No| IncStreak[Increment moves_since_progress]
    IncStreak --> HintCheck{3 or more moves<br/>without progress?}
    HintCheck -->|Yes| EscalateHint[Increase hint_level<br/>max level 3]
    HintCheck -->|No| ResetStreak
    EscalateHint --> UserPlays
    ResetStreak --> UserPlays

    UserPlays --> TimerEnd{Timer expired<br/>or user clicks Stop?}
    TimerEnd -->|No| UserPlays
    TimerEnd -->|Yes| ShowResults[Show results popup<br/>versus scenario targets]
    ShowResults --> Extend{Time expired and<br/>not yet extended?}
    Extend -->|Yes, user clicks plus 1 minute| ExtendTimer[Add 60s, resume countdown]
    ExtendTimer --> UserPlays
    Extend -->|No / already used| ResetTask[User clicks Reset]
    ResetTask --> FreeState
```

**Notes:**
- The open-ended planning task (scenario `"P"`) has no fixed Heat
  Exposure/green-fraction target — success is judged only on population
  (at least 70,000) and industry tile count (at least 3), with Heat Exposure
  minimized as a secondary objective.
- Hint escalation (`hint_level` 1 to 3) makes the LLM's suggestions
  progressively more concrete (Tip → Direction → Concrete), see
  `SCENARIO_HINTS` / `HINT_FRAMING` in `llm/prompt_builder.py`.
- The one-time plus-1-minute extension is only offered once per task attempt
  (`SessionState`/`PlanningTimerWidget._extended` flag).
