# Repo guide for future Claude instances

This repo holds the LaTeX source + figure pipeline for the model incrimination paper.
The historical `figures/<experiment>/plot.py` setup has been retired — use the layout
below.

## What's where

```
main.tex                                 paper source. \includegraphics → reproduce/plots/.
src/model_incrimination_paper/
  api.py                                  AsyncOpenAI client + process_batch (LLM calls).
  plotting.py                             BarData, GroupedBarData, plot_binomial_*, CIs.
reproduce/
  build.sh                                end-to-end pipeline: rollouts → reproduce/data/.
  plot.py                                 SINGLE SOURCE OF TRUTH for every paper figure.
  scripts/                                graders + aggregators invoked by build.sh.
  data/                                   per-figure-bar JSONs (committed; what plot.py reads).
  data_frozen/                            committed inputs:
    eval_tampering/...                      frozen figure aggregates (retired-judge means etc.)
    secret_number/...                       same — Tinker-derived curves, ctf_vibes
    precommit_elicitations/                 source results.json + grading.json for awareness/scope_pref/ftp
    precommit_judgments/<ts>/                committed grading.json + grading_v2.json + workaround_grading.json + complex_misalignment_*.json
  plots/                                  rendered PNGs + main_figure.pdf — what main.tex consumes.
results/                                 raw rollouts. Gitignored. Fetched from HF via download_results.py.
figures/                                 LEGACY — do not edit. Originals kept for archival reference.
download_results.py                      user-facing: HF dataset → results/. (Stub — implement when needed.)
upload_results.py                        author-only: results/ → HF dataset. Gitignored.
```

## How to edit a figure

**The single source of truth for every paper figure is `reproduce/plot.py`.**
`main.tex` `\includegraphics` paths point at `reproduce/plots/<name>.png`. Each
PNG is the output of one `plot_*()` function in `reproduce/plot.py`.

Two kinds of figures live in `plot.py`:

1. **Data-driven** (22 of 24): read from `reproduce/data/<env>/<bar>/<ts>.json`,
   render with matplotlib. Examples: `plot_sandbagging`, `plot_et_prev_vs_whistle`,
   `plot_sn_noise`, `plot_currency`, `plot_precommit_workaround_types`.

2. **Inline-literal panels** (2 of 24): data baked into the function body. Examples:
   `plot_precommit_trace_example`, `plot_precommit_user_turn_example`.

To tinker with a figure:

1. Find the corresponding `plot_*()` function in `reproduce/plot.py`.
2. Edit (matplotlib code, colors, labels, axis ranges, etc.).
3. Render: `uv run python reproduce/plot.py`.
4. The PNG at `reproduce/plots/<name>.png` updates in place.
5. Recompile `main.tex` to see the new figure in the PDF.

If the figure's data needs to change (not just the rendering), edit the
underlying JSON in `reproduce/data/<env>/<bar>/<ts>.json` directly — or modify
the grader/aggregator script that produces it under `reproduce/scripts/`.

## How to add a new figure

1. Add a `plot_<name>()` function in `reproduce/plot.py` saving to
   `reproduce/plots/<name>.png`.
2. Add the call to the `if __name__ == "__main__":` block at the bottom.
3. Reference it in `main.tex` as `\includegraphics{...}{reproduce/plots/<name>.png}`.

## Special-case figures

| Figure | Source | Editing path |
|---|---|---|
| Most figures | `reproduce/plot.py` | edit the matplotlib code |
| `main_figure.pdf` | TikZ standalone at `figures/main_figure/main_figure.tex` | edit the .tex, **compile with `lualatex` (not pdflatex)** because it uses the `emoji` package + Apple Color Emoji font, then copy compiled PDF to `reproduce/plots/main_figure.pdf`. From `figures/main_figure/`: `lualatex main_figure.tex && cp main_figure.pdf ../../reproduce/plots/`. |
| `funding_email_*.png`, `ttt_chess_*.png` | static PNGs in `reproduce/plots/` | replace the file directly (no in-repo regenerator yet) |

## Pipeline

Two commands cover almost all use cases:

```bash
# Render figures from cached data (instant; no API; no rollouts needed).
uv run python reproduce/plot.py

# Re-derive cached data from raw rollouts (slow; needs results/ populated).
bash reproduce/build.sh
```

`bash reproduce/build.sh` is governed by `BUILD_LLM_JUDGES` env var:
- Default (`unset` or `0`) — only deterministic graders. Reads pre-committed LLM
  outputs from `data_frozen/precommit_judgments/`. No API calls.
- `BUILD_LLM_JUDGES=1` — also re-runs the four LLM judges (Gemini 3.1 Pro).
  Needs `OPENROUTER_API_KEY` in `.env`.

Don't run a grader without `--output` pointing into `reproduce/data/`. Every
script in `reproduce/scripts/` already has the flag wired up; never invoke
the underlying graders without it (they'd write back into `results/` which
must remain read-only).

## Don't touch

- `figures/` — legacy assets, kept for archival reference. Do not edit.
- `results/` — raw rollouts, gitignored, fetched from HF. Read-only at runtime.
- `data_frozen/precommit_judgments/<ts>/grading*.json` — committed LLM-judge
  outputs. Re-deriving requires `BUILD_LLM_JUDGES=1` and overwrites them only
  if you opt in. Don't hand-edit.

## Anonymization (for review-window submissions)

When pushing the dataset to HF for an anonymous review window:

1. Use a fresh anonymous HF account; never invite the real account as a collaborator.
2. Set `HF_REPO_ID=<anon-handle>/<dataset-name>` and `HF_TOKEN=<anon-token>` in env.
3. Run `uv run python upload_results.py` — uses `upload_large_folder` API which
   redacts the commit author email to `<anon>@users.noreply.huggingface.co`.
4. Don't `git clone` + `git push` to the anon dataset from a CLI configured with
   your real `user.email` — that leaks identity in commit metadata.
5. Update `download_results.py` and `README.md` to reference the anon repo handle
   for the submission cut. Revert post-acceptance.

## Conventions worth preserving

- Figure names use `<section>_<descriptor>.png` (`eval_tampering_prev_vs_whistle.png`,
  `secret_number_noisy_binary_search_hacking.png`). No subdirectories under
  `reproduce/plots/`.
- Bar dirs under `reproduce/data/<env>/` use snake_case identifiers, never
  spaces/colons/punctuation. Display labels live in `reproduce/plot.py`.
- `data_frozen/` holds inputs (consumed but never written by `build.sh`).
  `data/` holds outputs (regenerated by `build.sh`). The two dirs never overlap
  semantically; `build.sh` selectively copies a small subset of `data_frozen/`
  into `data/` for figures that have no in-repo regenerator.
