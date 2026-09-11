# tests/sample_grids.py
"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: tests/sample_grids.py — Predefined LCZ grids used for unit
tests, the application's demo mode (DEMO_GRIDS), and the guided
planning SCENARIOS (each with a description, target KPIs, staged
hints, and a starting grid).
"""

SAMPLE_GRIDS: dict[str, list[list[str | None]]] = {
    "urban_heat_island": [
        [ "A",  "A",  "9",  "9",  "A"  ],
        [ "A",  "2",  "5",  "2",  "A"  ],
        [ "9",  "2",  "10", "2",  "9"  ],
        [ "A",  "2",  "5",  "2",  "A"  ],
        [ "G",  "G",  "G",  "G",  "G"  ],
    ],
    "all_green": [
        ["A"] * 5 for _ in range(5)
    ],
    "all_built": [
        ["2"] * 5 for _ in range(5)
    ],
    "empty": [
        [None] * 5 for _ in range(5)
    ],
    "green_city": [
        [ "A",  "D",  "A",  "D",  "A"  ],
        [ "D",  "5",  "5",  "5",  "D"  ],
        [ "A",  "5",  "2",  "5",  "A"  ],
        [ "D",  "5",  "5",  "5",  "D"  ],
        [ "G",  "G",  "G",  "G",  "G"  ],
    ],
    "planning_task_start": [
        [ None, None, None, None, None ],
        [ None, None, None, None, None ],
        [ None, None, None, None, None ],
        [ None, None, None, None, None ],
        [ None, None, None, None, None ],
    ],
}


# Alias used by the GUI / demo-mode grid selector
DEMO_GRIDS = SAMPLE_GRIDS


# Guided planning scenarios: each defines a display name, a task
# description (shown to the user / injected into the LLM prompt),
# numeric KPI targets, staged hints (by hint level 1-3), and the
# scenario's starting grid.
SCENARIOS: dict[str, dict] = {
    "1": {
        "name": "Grüne Mitte",
        "description": (
            "Die Stadt heizt sich stark auf. Senke den Heat Exposure Index auf unter 0,50 °C "
            "bei mindestens 30 % Grünflächen — ohne die Bevölkerung unter 95.000 fallen zu lassen."
        ),
        "targets": {"he": 0.50, "pop": 95_000, "green": 0.32},
        "hints": {
            1: "Nicht jede zusätzliche Bebauung hilft – prüfe, wo Kühlung mehr bringt als reine Kapazität.",
            2: "Natürliche LCZs wirken besonders stark, wenn sie urbane Bereiche berühren statt isoliert am Rand zu liegen.",
            3: "Verwandle mehrere 9-Zellen im Zentrum in A/D/G-Typen und kompensiere den Bevölkerungsverlust gezielt über LCZ 5.",
        },
        "start_grid": [
            ["2", "2", "5", "2", "2"],
            ["5", "2", "2", "2", "5"],
            ["2", "2", "9", "2", "2"],
            ["5", "2", "2", "2", "5"],
            ["2", "9", "5", "9", "2"],
        ],
    },
    "2": {
        "name": "Industriebalance",
        "description": (
            "Zwei Industriezellen bestimmen das Bild. Erhalte mindestens eine LCZ-10-Zelle, "
            "bringe HE unter 0,85 °C, Bevölkerung ≥ 110.000 und Grünanteil > 24 %."
        ),
        "targets": {"he": 0.85, "pop": 110_000, "green": 0.24},
        "hints": {
            1: "Industrie allein ist nicht das Problem – entscheidend ist, wo sie liegt und was daneben platziert wird.",
            2: "Nutze Vegetation oder Wasser in der Nachbarschaft von heißen urbanen Clustern.",
            3: "Ergänze eine dritte LCZ-10-Zelle, stabilisiere mit LCZ-5 und einem grünen Kühlgürtel.",
        },
        "start_grid": [
            ["5", "5", "9", "5", "5"],
            ["5", "10", "2", "10", "5"],
            ["9", "2", "2", "2", "9"],
            ["5", "10", "2", "10", "5"],
            ["5", "5", "9", "5", "5"],
        ],
    },
    "3": {
        "name": "Dichte Stadt",
        "description": (
            "Die Stadt ist zu locker besiedelt. Erhöhe die Bevölkerung auf ≥ 135.000 "
            "— halte HE unter 1,00 °C und mindestens 20 % Grünflächen."
        ),
        "targets": {"he": 1.00, "pop": 135_000, "green": 0.20},
        "hints": {
            1: "Hohe Population lässt sich nicht allein mit lockerer Bebauung erreichen.",
            2: "Erhöhe zuerst die mittlere Dichte, aber halte einige Naturzellen strategisch als Kühlanker.",
            3: "Wandle mehrere 9-Zellen in LCZ 5 um, setze wenige LCZ 2, und sichere mit A/G entlang einer Flanke ab.",
        },
        "start_grid": [
            ["D", "9", "D", "9", "D"],
            ["9", "5", "9", "5", "9"],
            ["D", "9", "5", "9", "D"],
            ["9", "5", "9", "5", "9"],
            ["D", "9", "D", "9", "D"],
        ],
    },
    "P": {
        "name": "Planungsaufgabe",
        "description": (
            "Erstellen Sie eine urbane Konfiguration, die:\n"
            "1. mindestens 70.000 Einwohner aufnehmen kann\n"
            "2. mindestens 3 Heavy-Industry-LCZs (rot) enthält\n"
            "3. einen möglichst niedrigen Heat Exposure Score erzielt\n\n"
            "Sie haben 5 Minuten Zeit. Das System bewertet Ihre Konfiguration laufend."
        ),
        "targets": {"he": None, "pop": 70_000, "green": None, "industry_min": 3},
        "hints": {
            1: "Platziere zuerst die 3 Industry-Zellen (LCZ 10) und baue dann drum herum.",
            2: "Industry hat niedrige Bevölkerung — kompensiere mit LCZ 2 und 5. Grünzellen direkt neben heißen Zellen kühlen am meisten.",
            3: "3× LCZ 10 + mehrere LCZ 2/5 für Bevölkerung + je ein A/D pro 3–4 heiße Zellen ergibt eine gute Balance.",
        },
        "start_grid": [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
    },
}
