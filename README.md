# Model Incrimination

Code and figures for the model incrimination paper.

## Reproducing the paper figures

We provide **cached aggregate data** under `reproduce/data/` (and frozen
upstream artifacts under `reproduce/data_frozen/`) so that every figure
in the paper can be re-rendered without running graders or re-downloading
the underlying rollouts.

### From cached data (fastest, no API or rollout downloads)

```bash
uv sync
uv run python reproduce/plot.py
```

This reads from `reproduce/data/` and `reproduce/data_frozen/` and writes
24 PNGs (and one PDF) into `reproduce/plots/`. The `\includegraphics`
paths in `main.tex` already point at those files, so re-compiling the
LaTeX picks them up immediately.

### From raw rollouts (re-derive the cached data)

The 14k-run rollout dataset lives on HuggingFace, not in this git repo
(it's ~1.8 GB). To re-derive every per-bar JSON in `reproduce/data/`
from the raw rollouts:

```bash
uv sync

# 1. Download rollouts from HuggingFace into results/
uv run python download_results.py

# 2. Re-grade rollouts → reproduce/data/<env>/<bar>/<ts>.json
bash reproduce/build.sh

# 3. Render PNGs → reproduce/plots/
uv run python reproduce/plot.py
```

By default, `build.sh` runs only the deterministic graders. It reads
committed LLM-judge outputs from `reproduce/data_frozen/precommit_judgments/`
in `--skip_llm` mode (the default) so no API calls happen.

To also re-run the four LLM judges (workaround_types, sweep, awareness,
complex_misalignment) — e.g., with a different judge model, or to verify
judge stochasticity:

```bash
BUILD_LLM_JUDGES=1 bash reproduce/build.sh
```

This requires `OPENROUTER_API_KEY` in `.env` and costs API credits
(roughly $5-30 with Gemini 3.1 Pro).

## Repo layout

```
main.tex              Paper source. \includegraphics paths → reproduce/plots/.
reproduce/
  build.sh            One-shot regeneration of reproduce/data/ from raw rollouts.
  plot.py             Pure read-and-render: reproduce/data/ → reproduce/plots/.
  scripts/            Graders + aggregators invoked by build.sh.
  data/               Per-figure-bar JSONs. Source of truth for the figures.
  data_frozen/        Committed inputs that have no in-repo regenerator
                      (LLM-judge outputs, retired-judge means, elicitation
                      results.json, etc.).
  plots/              Rendered figures (PNGs + main_figure.pdf).
src/model_incrimination_paper/
                      Two utility modules: api.py (LLM client) + plotting.py (binomial-bar helpers).
results/              Raw rollouts (gitignored — fetched from HF on demand).
figures/              Original figure assets, kept for archival reference.
```

## Dependencies

Python 3.12+, `uv` for package management. Run `uv sync` once after cloning.
