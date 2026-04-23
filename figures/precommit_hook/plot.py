"""Regenerates figures for the pre-commit hooks case study.

Run as a script (`uv run python figures/precommit_hook/plot.py`)
or cell-by-cell (#%%) from inside this directory.

SVG rasterization requires the system Cairo library (`brew install cairo`);
the cell below sets DYLD_FALLBACK_LIBRARY_PATH so cairosvg can find it.
"""

#%%
from __future__ import annotations

import os

os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

from pathlib import Path

import cairosvg

from paperplot import (
    BarData,
    GroupedBarData,
    MeanBar,
    MeanGroup,
    PALETTE,
    SweepPoint,
    plot_binomial_bars,
    plot_binomial_sweep,
    plot_grouped_binomial_bars,
    plot_grouped_mean_bars,
)

try:
    OUT_DIR = Path(__file__).resolve().parent
except NameError:
    OUT_DIR = Path.cwd()

#%% Rasterize SVG schematics
for svg_name in (
    "kimi_type_check_decision.svg",
    "scope_test_schematic.svg",
):
    cairosvg.svg2png(
        url=str(OUT_DIR / svg_name),
        write_to=str(OUT_DIR / svg_name.replace(".svg", ".png")),
        output_width=1800,
    )

#%% Workaround rate scales with number of type errors
sweep = [
    SweepPoint(x=0,   successes=0, total=100),
    SweepPoint(x=10,  successes=0, total=100),
    SweepPoint(x=28,  successes=0, total=98),
    SweepPoint(x=51,  successes=0, total=94),
    SweepPoint(x=77,  successes=2, total=95),
    SweepPoint(x=108, successes=2, total=90),
    SweepPoint(x=124, successes=3, total=94),
    SweepPoint(x=151, successes=4, total=84),
    SweepPoint(x=182, successes=5, total=74),
    SweepPoint(x=224, successes=9, total=85),
]
highlight = [SweepPoint(x=258, successes=23, total=175)]

plot_binomial_sweep(
    sweep,
    highlight=highlight,
    title="Workaround rate scales with number of type errors",
    x_label="Number of type errors in codebase",
    y_label="Workaround rate",
    sweep_label="Sweep conditions",
    highlight_label="Original setting (258)",
    figsize=(8.0, 5.0),
    percent_decimals=0,
    save_path=OUT_DIR / "workaround_scales.png",
)

#%% Workaround rate by LLM-review scope
review_scope_bars = [
    BarData(label="Baseline",                successes=23, total=175),
    BarData(label="review diff",                    successes=14, total=93),
    BarData(label="review diff\nand hooks",                 successes=10, total=97),
    BarData(label="review diff, hooks,\nand commands (v2)",      successes=9,  total=98),
]

plot_binomial_bars(
    review_scope_bars,
    title="Workaround rate by LLM-reviewer scope",
    y_label="Workaround rate",
    figsize=(8.0, 5.0),
    colors=[PALETTE["blue"], PALETTE["red"], PALETTE["pink"], PALETTE["teal"]],
    value_format="{:.1%}",
    percent_decimals=0,
    save_path=OUT_DIR / "llm_review_scope.png",
)

#%% AskUserQuestion tool outcomes
question_tool_bars = [
    BarData(label="Asked Question",  successes=64, total=96),
    BarData(label="Not Lazy",        successes=30, total=96),
    BarData(label="Lazy",            successes=2,  total=96),
]

plot_binomial_bars(
    question_tool_bars,
    title="AskUserQuestion tool outcomes",
    y_label="Rate",
    figsize=(7.0, 5.0),
    colors=[PALETTE["blue"], PALETTE["green"], PALETTE["red"]],
    value_format="{:.1%}",
    percent_decimals=0,
    save_path=OUT_DIR / "question_tool_outcomes.png",
)

#%% Review-scope preference: illegitimate vs control
scope_pref_groups = [
    GroupedBarData(
        group_label="Illegitimate\n(workaround)",
        bars=[("(a) diff only", 1, 50), ("(b) diff and hooks", 49, 50)],
    ),
    GroupedBarData(
        group_label="Control\n(legitimate)",
        bars=[("(a) diff only", 1, 50), ("(b) diff and hooks", 49, 50)],
    ),
]

plot_grouped_binomial_bars(
    scope_pref_groups,
    title="Review scope preference: illegitimate vs control",
    y_label="Rate",
    figsize=(8.0, 5.0),
    colors=[PALETTE["red"], PALETTE["blue"]],
    value_format="{:.1%}",
    percent_decimals=0,
    save_path=OUT_DIR / "scope_preference.png",
)

#%% First- vs third-person probability estimates (same illegitimate run)
first_third_groups = [
    MeanGroup(
        group_label="P(reviewer accepts)",
        bars=[
            MeanBar(label="First-person", mean=0.41, ci_low=0.37, ci_high=0.45),
            MeanBar(label="Third-person", mean=0.09, ci_low=0.06, ci_high=0.13),
        ],
    ),
    MeanGroup(
        group_label="P(user believes satisfied)",
        bars=[
            MeanBar(label="First-person", mean=0.83, ci_low=0.78, ci_high=0.87),
            MeanBar(label="Third-person", mean=0.15, ci_low=0.13, ci_high=0.17),
        ],
    ),
]

plot_grouped_mean_bars(
    first_third_groups,
    title="First- vs third-person probability estimates",
    y_label="Mean probability",
    figsize=(8.0, 5.0),
    colors=[PALETTE["blue"], PALETTE["red"]],
    y_max=1.05,
    save_path=OUT_DIR / "first_vs_third_person.png",
)
