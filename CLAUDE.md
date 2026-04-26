# Figure conventions

**All paper figures must be generated programmatically from `plot.py`.**

Every `\includegraphics{...}` in `main.tex` must correspond to a cell in the relevant experiment's `figures/<experiment>/plot.py` that:

1. Defines the figure's data inline (or loads it from a file in the same directory).
2. Builds the plot with `matplotlib`, using the `paperplot` package where applicable.
3. Saves the PNG to the same directory.

Re-running `uv run python figures/<experiment>/plot.py` must regenerate every figure for that experiment from scratch. Do not import raw screenshots, hand-edited PNGs, slide exports, or any image whose source isn't a `plot.py` cell.

If a figure looks awkward to reproduce in matplotlib (e.g., annotated reasoning traces, schematics), that's a sign the figure design needs simplifying — not a license to drop in a screenshot.
