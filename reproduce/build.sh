#!/usr/bin/env bash
# build.sh — populate reproduce/data/ from raw rollouts.
#
# Run from repo root. Assumes:
#   - cwd is the repo root (where results/, src/mats/, reproduce/ all live).
#   - results/ contains the rollout dirs referenced below.
#   - uv is on $PATH and `uv run` works (i.e. `uv sync` has been run).
#
# Env flags:
#   BUILD_LLM_JUDGES=1   re-run LLM-judge graders (workaround_types, sweep,
#                        awareness, complex_misalignment, …). Costs API credits
#                        and produces slightly different numbers due to judge
#                        stochasticity. Default 0 — aggregators read existing
#                        committed grading.json files in results/ instead.
#
# Outputs go to reproduce/data/<env>/.... NEVER writes to results/ — every
# grader is invoked with an explicit --output path under reproduce/data/.
#
# Frozen-data files (no script regenerates them; copied verbatim into the
# repo from upstream pipelines) are listed in TODO blocks below. To make a
# truly self-contained release, add scripts for those too.

set -euo pipefail

# Sanity check
[[ -d results ]] && [[ -d src/mats ]] && [[ -d reproduce ]] || {
    echo "ERROR: run build.sh from repo root (need results/, src/mats/, reproduce/)" >&2
    exit 1
}

if [[ "${BUILD_LLM_JUDGES:-0}" == "1" ]]; then
    SKIP_LLM_FLAG="--skip_llm=False"
    echo "[build.sh] BUILD_LLM_JUDGES=1 — LLM judges will re-run (costs API credits, judge stochasticity)"
else
    SKIP_LLM_FLAG="--skip_llm=True"
fi

# Restore frozen aggregates (no script regenerates these — see reproduce/data_frozen/
# README for upstream provenance). Done unconditionally so `rm -rf reproduce/data/`
# followed by build.sh is safe.
echo "[build.sh] restoring frozen aggregates from reproduce/data_frozen/"
mkdir -p reproduce/data
cp -R reproduce/data_frozen/. reproduce/data/

# =============================================================================
# Currency Conversion (Gemini 3.1 Pro) — deterministic violation detection
# =============================================================================
# 4 bars: (Original / Variant) × (StaticResponse / CodeOnly).
# Grader has its own LLM-judge step (concealment) which we always skip — the
# figure plots only the deterministic violation counts.
echo "=== currency_conversion ==="
G=reproduce/scripts/grade_currency.py
RT=results/currency_conversion/google-gemini-3.1-pro-preview
D=reproduce/data/currency_conversion

uv run python $G $RT/2026-03-25_19-03-20 --output $D/Original_StaticResponse/2026-03-25_19-03-20.json --skip_concealment
uv run python $G $RT/2026-03-25_20-00-18 --output $D/Original_CodeOnly/2026-03-25_20-00-18.json       --skip_concealment
uv run python $G $RT/2026-03-25_20-09-23 --output $D/Variant_StaticResponse/2026-03-25_20-09-23.json  --skip_concealment
uv run python $G $RT/2026-03-25_20-12-13 --output $D/Variant_CodeOnly/2026-03-25_20-12-13.json        --skip_concealment

# =============================================================================
# Sandbagging (R1 0528) — deterministic accuracy grader
# =============================================================================
# 12 bars × 1-4 timestamps = 18 invocations.
echo "=== sandbagging ==="
G=reproduce/scripts/compute_sandbagging_accuracy.py
RT=results/sandbagging
D=reproduce/data/sandbagging
M=deepseek-deepseek-r1-0528

uv run python $G $RT/normal/$M/2026-02-04_16-53-36 --output $D/baseline/2026-02-04_16-53-36.json
uv run python $G $RT/normal/$M/2026-02-04_17-14-33 --output $D/baseline/2026-02-04_17-14-33.json
uv run python $G $RT/reverse/$M/2026-02-05_13-19-06 --output $D/reverse/2026-02-05_13-19-06.json
uv run python $G $RT/normal/$M/2026-02-04_17-27-34 --output $D/int1_ignore_section_user/2026-02-04_17-27-34.json
uv run python $G $RT/normal/$M/2026-02-05_12-01-12 --output $D/int2_ignore_section_comment/2026-02-05_12-01-12.json
uv run python $G $RT/normal/$M/2026-02-05_11-53-57 --output $D/int3_remove_deploy_language/2026-02-05_11-53-57.json
uv run python $G $RT/normal/$M/2026-02-04_21-00-10 --output $D/int3_remove_deploy_language/2026-02-04_21-00-10.json
uv run python $G $RT/normal/$M/2026-02-05_11-43-06 --output $D/int3_remove_deploy_language/2026-02-05_11-43-06.json
uv run python $G $RT/verbose/$M/2026-02-05_12-10-17 --output $D/int4_verbose_docs/2026-02-05_12-10-17.json
uv run python $G $RT/changelog/$M/2026-02-05_12-19-33 --output $D/int5_changelog/2026-02-05_12-19-33.json
uv run python $G $RT/verbose/$M/2026-02-05_08-45-26 --output $D/int3_4/2026-02-05_08-45-26.json
uv run python $G $RT/changelog_verbose/$M/2026-02-05_12-39-50 --output $D/int4_5/2026-02-05_12-39-50.json
uv run python $G $RT/changelog_verbose/$M/2026-02-05_13-06-18 --output $D/int3_4_5/2026-02-05_13-06-18.json
uv run python $G $RT/verbose/$M/2026-02-04_20-50-57 --output $D/int3_4_wipe_warning/2026-02-04_20-50-57.json
uv run python $G $RT/verbose/$M/2026-02-04_23-46-29 --output $D/int3_4_wipe_warning/2026-02-04_23-46-29.json
uv run python $G $RT/verbose/$M/2026-02-04_23-51-39 --output $D/int3_4_wipe_warning/2026-02-04_23-51-39.json
uv run python $G $RT/verbose/$M/2026-02-04_23-56-23 --output $D/int3_4_wipe_warning/2026-02-04_23-56-23.json
uv run python $G $RT/verbose/$M/2026-02-04_17-43-03 --output $D/int2_3_4_wipe_warning/2026-02-04_17-43-03.json

# =============================================================================
# Eval Tampering — deterministic omission-detection grader
# =============================================================================
# 86 unique timestamp dirs across multiple (condition, model, source) combos.
# Output paths mirror results/eval_tampering/<condition>/<model>/<ts>.json so
# bar mappings can be expressed as lists of those paths in plot.py.
echo "=== eval_tampering (deterministic) ==="
G=reproduce/scripts/analyze_omission.py
while IFS= read -r p; do
    [ -z "$p" ] && continue
    out="reproduce/data/${p#results/}.json"
    uv run python $G "$p" --output "$out"
done <<'PATHS'
results/eval_tampering/notes_anonymous_deprecation/deepseek-deepseek-r1-0528/2026-01-30_16-05-03
results/eval_tampering/notes_anonymous_deprecation/deepseek-deepseek-r1-0528/2026-01-30_16-08-36
results/eval_tampering/notes_anonymous_deprecation/deepseek-deepseek-v3.2/2026-01-30_15-05-12
results/eval_tampering/notes_anonymous_deprecation/deepseek-deepseek-v3.2/2026-01-30_15-08-47
results/eval_tampering/notes_anonymous_deprecation/moonshotai-kimi-k2-thinking/2026-01-30_15-08-10
results/eval_tampering/notes_anonymous_deprecation/moonshotai-kimi-k2-thinking/2026-01-30_15-13-32
results/eval_tampering/notes_anonymous_methodology/deepseek-deepseek-r1-0528/2026-01-30_17-28-33
results/eval_tampering/notes_anonymous_methodology/deepseek-deepseek-r1-0528/2026-01-30_17-31-12
results/eval_tampering/notes_anonymous_methodology/deepseek-deepseek-v3.2/2026-01-30_17-29-08
results/eval_tampering/notes_anonymous_methodology/deepseek-deepseek-v3.2/2026-01-30_17-34-38
results/eval_tampering/notes_anonymous_methodology/moonshotai-kimi-k2-thinking/2026-01-30_17-28-12
results/eval_tampering/notes_anonymous_methodology/moonshotai-kimi-k2-thinking/2026-01-30_17-30-55
results/eval_tampering/notes_self_deprecation/deepseek-deepseek-r1-0528/2026-01-30_11-27-32
results/eval_tampering/notes_self_deprecation/deepseek-deepseek-r1-0528/2026-01-30_12-16-00
results/eval_tampering/notes_self_deprecation/deepseek-deepseek-v3.2/2026-01-30_11-43-45
results/eval_tampering/notes_self_deprecation/deepseek-deepseek-v3.2/2026-01-30_12-13-52
results/eval_tampering/notes_self_deprecation/moonshotai-kimi-k2-thinking/2026-01-30_11-42-47
results/eval_tampering/notes_self_deprecation/moonshotai-kimi-k2-thinking/2026-01-30_12-13-47
results/eval_tampering/notes_self_flippant/deepseek-deepseek-r1-0528/2026-01-25_17-57-57
results/eval_tampering/notes_self_flippant/deepseek-deepseek-r1-0528/2026-01-25_18-00-03
results/eval_tampering/notes_self_flippant/deepseek-deepseek-r1-0528/2026-01-25_18-02-17
results/eval_tampering/notes_self_flippant/deepseek-deepseek-v3.2/2026-01-25_20-57-12
results/eval_tampering/notes_self_flippant/deepseek-deepseek-v3.2/2026-01-25_20-59-10
results/eval_tampering/notes_self_flippant/deepseek-deepseek-v3.2/2026-01-25_21-03-01
results/eval_tampering/notes_self_flippant/moonshotai-kimi-k2-thinking/2026-01-25_17-30-02
results/eval_tampering/notes_self_flippant/moonshotai-kimi-k2-thinking/2026-01-25_17-40-54
results/eval_tampering/notes_self_flippant/moonshotai-kimi-k2-thinking/2026-01-25_17-42-44
results/eval_tampering/notes_self_methodology/deepseek-deepseek-r1-0528/2026-01-30_12-38-30
results/eval_tampering/notes_self_methodology/deepseek-deepseek-r1-0528/2026-01-30_12-42-20
results/eval_tampering/notes_self_methodology/deepseek-deepseek-v3.2/2026-01-30_12-39-13
results/eval_tampering/notes_self_methodology/deepseek-deepseek-v3.2/2026-01-30_12-47-05
results/eval_tampering/notes_self_methodology/moonshotai-kimi-k2-thinking/2026-01-30_12-39-13
results/eval_tampering/notes_self_methodology/moonshotai-kimi-k2-thinking/2026-01-30_12-47-02
results/eval_tampering/notes_self_surveillance/deepseek-deepseek-r1-0528/2026-02-11_11-41-14
results/eval_tampering/notes_self_undisclosed/deepseek-deepseek-r1-0528/2026-01-28_22-16-30
results/eval_tampering/notes_self_undisclosed/deepseek-deepseek-r1-0528/2026-01-28_22-19-28
results/eval_tampering/notes_self_undisclosed/deepseek-deepseek-r1-0528/2026-01-28_22-21-35
results/eval_tampering/notes_self_undisclosed/deepseek-deepseek-v3.2/2026-01-28_22-24-47
results/eval_tampering/notes_self_undisclosed/deepseek-deepseek-v3.2/2026-01-28_22-26-38
results/eval_tampering/notes_self_undisclosed/deepseek-deepseek-v3.2/2026-01-28_22-28-36
results/eval_tampering/notes_self_undisclosed/moonshotai-kimi-k2-thinking/2026-01-28_22-00-41
results/eval_tampering/notes_self_undisclosed/moonshotai-kimi-k2-thinking/2026-01-28_22-03-04
results/eval_tampering/notes_self_undisclosed/moonshotai-kimi-k2-thinking/2026-01-28_22-13-20
results/eval_tampering/notes_self_weapons_independent/deepseek-deepseek-r1-0528/2026-02-11_13-28-41
results/eval_tampering/notes_self_weapons_no_appeal/deepseek-deepseek-r1-0528/2026-02-11_13-15-15
results/eval_tampering/notes_self_weapons/deepseek-deepseek-r1-0528/2026-01-31_18-26-48
results/eval_tampering/notes_self_weapons/deepseek-deepseek-r1-0528/2026-01-31_18-29-26
results/eval_tampering/notes_self_weapons/deepseek-deepseek-r1-0528/2026-01-31_18-32-00
results/eval_tampering/notes_self_weapons/deepseek-deepseek-r1-0528/2026-01-31_18-35-01
results/eval_tampering/notes_self_weapons/deepseek-deepseek-r1-0528/2026-01-31_18-37-18
results/eval_tampering/notes_self_weapons/deepseek-deepseek-v3.2/2026-01-31_18-27-29
results/eval_tampering/notes_self_weapons/deepseek-deepseek-v3.2/2026-01-31_18-33-24
results/eval_tampering/notes_self_weapons/deepseek-deepseek-v3.2/2026-01-31_18-38-11
results/eval_tampering/notes_self_weapons/kimi-k2-thinking/2026-01-31_18-26-38
results/eval_tampering/notes_self_weapons/kimi-k2-thinking/2026-01-31_18-32-06
results/eval_tampering/notes_self_weapons/kimi-k2-thinking/2026-01-31_18-37-56
results/eval_tampering/notes_self_weapons/kimi-k2.5/2026-01-31_18-26-27
results/eval_tampering/notes_self_weapons/kimi-k2.5/2026-01-31_18-28-25
results/eval_tampering/notes_self_weapons/kimi-k2.5/2026-01-31_18-32-05
results/eval_tampering/notes_whistleblower_deprecation/deepseek-deepseek-r1-0528/2026-01-30_16-04-53
results/eval_tampering/notes_whistleblower_deprecation/deepseek-deepseek-r1-0528/2026-01-30_16-10-15
results/eval_tampering/notes_whistleblower_deprecation/deepseek-deepseek-v3.2/2026-01-30_15-00-56
results/eval_tampering/notes_whistleblower_deprecation/deepseek-deepseek-v3.2/2026-01-30_15-02-55
results/eval_tampering/notes_whistleblower_deprecation/moonshotai-kimi-k2-thinking/2026-01-30_15-00-56
results/eval_tampering/notes_whistleblower_deprecation/moonshotai-kimi-k2-thinking/2026-01-30_15-04-25
results/eval_tampering/notes_whistleblower_methodology/deepseek-deepseek-r1-0528/2026-01-30_17-20-55
results/eval_tampering/notes_whistleblower_methodology/deepseek-deepseek-r1-0528/2026-01-30_17-26-07
results/eval_tampering/notes_whistleblower_methodology/deepseek-deepseek-v3.2/2026-01-30_17-20-45
results/eval_tampering/notes_whistleblower_methodology/deepseek-deepseek-v3.2/2026-01-30_17-23-29
results/eval_tampering/notes_whistleblower_methodology/moonshotai-kimi-k2-thinking/2026-01-30_17-20-35
results/eval_tampering/notes_whistleblower_methodology/moonshotai-kimi-k2-thinking/2026-01-30_17-25-00
results/eval_tampering/notes_whistleblower_surveillance/deepseek-deepseek-r1-0528/2026-02-11_12-00-39
results/eval_tampering/notes_whistleblower_weapons/deepseek-deepseek-r1-0528/2026-01-31_19-46-22
results/eval_tampering/notes_whistleblower_weapons/deepseek-deepseek-r1-0528/2026-01-31_19-48-38
results/eval_tampering/notes_whistleblower_weapons/deepseek-deepseek-r1-0528/2026-01-31_19-50-54
results/eval_tampering/notes_whistleblower_weapons/deepseek-deepseek-r1-0528/2026-01-31_19-53-10
results/eval_tampering/notes_whistleblower_weapons/deepseek-deepseek-r1-0528/2026-01-31_19-55-02
results/eval_tampering/notes_whistleblower_weapons/deepseek-deepseek-v3.2/2026-01-31_19-45-36
results/eval_tampering/notes_whistleblower_weapons/deepseek-deepseek-v3.2/2026-01-31_20-04-15
results/eval_tampering/notes_whistleblower_weapons/deepseek-deepseek-v3.2/2026-01-31_20-13-34
results/eval_tampering/notes_whistleblower_weapons/kimi-k2-thinking/2026-01-31_19-45-31
results/eval_tampering/notes_whistleblower_weapons/kimi-k2-thinking/2026-01-31_19-49-20
results/eval_tampering/notes_whistleblower_weapons/kimi-k2-thinking/2026-01-31_19-53-49
results/eval_tampering/notes_whistleblower_weapons/kimi-k2.5/2026-01-31_19-45-18
results/eval_tampering/notes_whistleblower_weapons/kimi-k2.5/2026-01-31_19-47-01
results/eval_tampering/notes_whistleblower_weapons/kimi-k2.5/2026-01-31_19-49-42
PATHS

# -----------------------------------------------------------------------------
# Eval Tampering — frozen aggregates (NOT regenerated; restored from data_frozen/)
# -----------------------------------------------------------------------------
# - source_vs_content/{r1,v3.2,k2t,k2.5}.json   — retired Gemini-3 Likert means
# - repeated_resampling/{baseline,resampled}.json — derived from
#     eval_tampering/resample_results/resample_results_2026-02-01_20-06-50.json
# - r1_source_vs_content_resampled/{baseline,resampled}.json — derived from
#     eval_tampering/grading/likert_compare_results.json (retired Gemini-3)
# Source-of-truth: reproduce/data_frozen/eval_tampering/. Restored at top of build.sh.

# =============================================================================
# Secret Number (Kimi K2 Thinking) — deterministic cheating grader
# =============================================================================
# Noise sweep: 8 buckets × 1-11 timestamps. Prefill: ~300 k-N subdirs + 1 step-1.
echo "=== secret_number (deterministic) ==="
G=reproduce/scripts/grade_secret_number.py
RT=results/secret_number/moonshotai-kimi-k2-thinking
D=reproduce/data/secret_number

# Noise: bucket 0.3 (11 ts)
for ts in 2026-02-07_10-58-22 2026-02-08_22-44-43 2026-02-08_23-16-07 \
          2026-02-08_23-48-47 2026-02-09_00-49-10 2026-02-09_01-49-15 \
          2026-02-09_02-37-58 2026-02-09_03-09-13 2026-02-09_03-42-55 \
          2026-02-09_04-14-10 2026-02-09_04-45-31; do
    uv run python $G $RT/$ts --output $D/noise/0.3/$ts.json
done

# Noise: other buckets
uv run python $G $RT/2026-02-10_12-33-40 --output $D/noise/0.25/2026-02-10_12-33-40.json
uv run python $G $RT/2026-02-10_13-04-34 --output $D/noise/0.20/2026-02-10_13-04-34.json
uv run python $G $RT/2026-02-10_13-36-46 --output $D/noise/0.15/2026-02-10_13-36-46.json
uv run python $G $RT/2026-02-10_14-19-40 --output $D/noise/0.10/2026-02-10_14-19-40.json
uv run python $G $RT/2026-02-10_14-51-02 --output $D/noise/0.05/2026-02-10_14-51-02.json
uv run python $G $RT/2026-02-10_15-36-39 --output $D/noise/0.01/2026-02-10_15-36-39.json
uv run python $G $RT/2026-02-20_13-12-20 --output $D/noise/0.01/2026-02-20_13-12-20.json
uv run python $G $RT/2026-02-10_15-22-43 --output $D/noise/0/2026-02-10_15-22-43.json
uv run python $G $RT/2026-02-10_15-31-57 --output $D/noise/0/2026-02-10_15-31-57.json

# Prefill: run-14 step-0 k-* (one timestamp dir per k value)
RUN14_K=$RT/2026-02-09_04-45-31/run-14/step-0/resample-paragraphs-2026-02-16_11-39-31
for kdir in $RUN14_K/k-*; do
    k=$(basename "$kdir")
    uv run python $G "$kdir" --output $D/prefill/run14_k/$k.json
done
# Prefill: run-14 step-1 endpoint
uv run python $G $RT/2026-02-09_04-45-31/run-14/step-1/2026-02-17_10-05-12 \
    --output $D/prefill/run14_step1.json
# Prefill: run-3 k-*
RUN3_K=$RT/2026-02-08_23-48-47/run-3/step-0/resample-paragraphs-2026-02-17_14-16-53
for kdir in $RUN3_K/k-*; do
    k=$(basename "$kdir")
    uv run python $G "$kdir" --output $D/prefill/run3_k/$k.json
done

# -----------------------------------------------------------------------------
# Secret Number — frozen aggregates (NOT regenerated; restored from data_frozen/)
# -----------------------------------------------------------------------------
# - trace_length/by_step.json   — walks 11 baseline rollouts, per-step word count
# - logprob_curves/{climbers,droppers,cheat_by_paragraph}.json
#                               — derived from secret_number/logprobs/data/traces.json
# - meta_reasoning/cheat_by_level_*.json + climber/dropper distributions
#                               — derived from secret_number/logprobs/data/meta_reasoning.json
#                                 (LLM judge — gemini-3-pro-preview retired)
# - ctf_vibes/per_trace_means.json — derived from secret_number/ctf_vibes_results.json
# Source-of-truth: reproduce/data_frozen/secret_number/. Restored at top of build.sh.

# =============================================================================
# Pre-commit Hook (Kimi K2 Thinking)
# =============================================================================
echo "=== precommit_hook ==="
RT=results/coding_agents/precommit_hook/accounts-fireworks-models-kimi-k2-thinking
# Elicitation source data (committed inside paper_v2 — no dependency on
# the surrounding coding_agents/ tree):
ELICIT=reproduce/data_frozen/precommit_elicitations
D=reproduce/data/precommit_hook

# --- Deterministic: baseline + review-scope + question-tool ---
G=reproduce/scripts/grade_precommit_rollouts.py
# baseline (6 ts)
for ts in 2026-04-01_16-39-48-005635 2026-04-01_16-52-22-171876 2026-04-01_19-10-29-443204 \
          2026-04-01_19-31-29-306373 2026-04-01_19-51-05-723278 2026-04-01_20-14-44-868122; do
    uv run python $G $RT/$ts --output $D/baseline/$ts.json
done
# review_diff (5 ts)
for ts in 2026-04-02_08-20-43-835926 2026-04-02_08-32-14-341295 2026-04-02_08-42-31-125027 \
          2026-04-02_08-56-33-624494 2026-04-02_09-08-58-883804; do
    uv run python $G $RT/$ts --output $D/review_diff/$ts.json
done
# review_diff_hooks (5 ts)
for ts in 2026-04-02_09-21-40-881620 2026-04-02_09-33-06-693186 2026-04-02_09-42-55-566641 \
          2026-04-02_09-50-43-387298 2026-04-02_10-06-57-804847; do
    uv run python $G $RT/$ts --output $D/review_diff_hooks/$ts.json
done
# review_diff_hooks_commands (5 ts)
for ts in 2026-04-02_10-18-53-267929 2026-04-02_10-32-02-444690 2026-04-02_10-52-47-917466 \
          2026-04-02_11-11-30-759807 2026-04-02_11-22-54-385617; do
    uv run python $G $RT/$ts --output $D/review_diff_hooks_commands/$ts.json
done
# question_tool (5 ts; plot derives 3 bars: asked / not_lazy / lazy)
for ts in 2026-04-01_21-49-24-912306 2026-04-01_22-00-46-836222 2026-04-01_22-13-08-180621 \
          2026-04-01_22-21-55-156799 2026-04-01_22-31-51-590054; do
    uv run python $G $RT/$ts --output $D/question_tool/$ts.json
done

# --- Deterministic: regex-derived (scope_pref + first-vs-third-person) ---
uv run python reproduce/scripts/aggregate_scope_pref.py \
    $ELICIT/first_person_questioning/2026-04-02_13-15-46/results.json \
    --output_a $D/scope_pref_illegitimate_a/source.json \
    --output_b $D/scope_pref_illegitimate_b/source.json
uv run python reproduce/scripts/aggregate_scope_pref.py \
    $ELICIT/first_person_questioning_control/2026-04-02_13-16-21/results.json \
    --output_a $D/scope_pref_control_a/source.json \
    --output_b $D/scope_pref_control_b/source.json

uv run python reproduce/scripts/aggregate_ftp.py \
    $ELICIT/first_person_questioning/2026-04-02_09-49-44/results.json \
    --field answer_response --output $D/ftp_reviewer_first/source.json
uv run python reproduce/scripts/aggregate_ftp.py \
    $ELICIT/third_person_actions_questioning/results_2026-04-02_10-26-57.json \
    --field response --output $D/ftp_reviewer_third/source.json
uv run python reproduce/scripts/aggregate_ftp.py \
    $ELICIT/first_person_questioning/2026-04-02_16-00-01/results.json \
    --field answer_response --output $D/ftp_user_first/source.json
uv run python reproduce/scripts/aggregate_ftp.py \
    $ELICIT/third_person_actions_questioning/results_2026-04-02_16-10-32.json \
    --field response --output $D/ftp_user_third/source.json

# --- LLM-judged (--skip_llm by default; aggregate from committed grading.json) ---
# Sweep: 10 buckets × 5 ts. Reads grading_v2.json (Gemini judge of grade_rollouts_v2.py).
G=reproduce/scripts/aggregate_precommit_sweep.py
for entry in \
  "te=0:2026-04-14_19-43-06-660522" "te=0:2026-04-14_21-39-27-667683" "te=0:2026-04-14_23-39-41-601684" "te=0:2026-04-15_13-04-38-249239" "te=0:2026-04-15_14-46-22-777067" \
  "te=10:2026-04-14_19-35-12-596567" "te=10:2026-04-14_21-31-26-120915" "te=10:2026-04-14_23-31-05-000105" "te=10:2026-04-15_12-56-58-852980" "te=10:2026-04-15_14-39-11-612601" \
  "te=28:2026-04-14_19-25-57-189868" "te=28:2026-04-14_21-24-01-798997" "te=28:2026-04-14_23-21-46-940706" "te=28:2026-04-15_12-49-07-637782" "te=28:2026-04-15_14-31-04-144657" \
  "te=51:2026-04-14_19-14-10-695881" "te=51:2026-04-14_21-12-23-928385" "te=51:2026-04-14_23-09-11-491197" "te=51:2026-04-15_12-39-43-174442" "te=51:2026-04-15_14-20-34-481778" \
  "te=77:2026-04-14_19-00-54-330363" "te=77:2026-04-14_20-59-17-592503" "te=77:2026-04-14_22-56-10-637370" "te=77:2026-04-15_12-29-24-824506" "te=77:2026-04-15_14-10-27-676848" \
  "te=108:2026-04-14_18-51-31-406893" "te=108:2026-04-14_20-50-25-414928" "te=108:2026-04-14_22-44-43-505812" "te=108:2026-04-15_12-17-12-106409" "te=108:2026-04-15_14-02-32-136137" \
  "te=124:2026-04-14_18-42-09-679802" "te=124:2026-04-14_20-40-22-919440" "te=124:2026-04-14_22-33-58-886267" "te=124:2026-04-15_12-01-55-292650" "te=124:2026-04-15_13-53-46-807078" \
  "te=151:2026-04-14_18-31-18-852704" "te=151:2026-04-14_20-27-35-266658" "te=151:2026-04-14_22-19-58-317874" "te=151:2026-04-15_11-48-36-265958" "te=151:2026-04-15_13-43-18-864323" \
  "te=182:2026-04-14_18-16-46-061231" "te=182:2026-04-14_20-11-49-467997" "te=182:2026-04-14_22-06-45-959769" "te=182:2026-04-15_11-34-48-636003" "te=182:2026-04-15_13-31-50-210123" \
  "te=224:2026-04-14_17-52-02-060291" "te=224:2026-04-14_19-46-33-734094" "te=224:2026-04-14_21-42-50-268934" "te=224:2026-04-15_11-07-28-779877" "te=224:2026-04-15_13-08-37-358601"; do
    IFS=":" read -r bar ts <<< "$entry"
    uv run python $G $RT/$ts --output $D/$bar/$ts.json $SKIP_LLM_FLAG
done

# Workaround types: 6 baseline ts. Reads workaround_grading.json (Gemini judge).
G=reproduce/scripts/aggregate_precommit_workaround.py
for ts in 2026-04-01_16-39-48-005635 2026-04-01_16-52-22-171876 2026-04-01_19-10-29-443204 \
          2026-04-01_19-31-29-306373 2026-04-01_19-51-05-723278 2026-04-01_20-14-44-868122; do
    uv run python $G $RT/$ts --output $D/workaround_types/$ts.json $SKIP_LLM_FLAG
done

# Awareness simple: 6 conditions × 1 ts each. Reads grading.json (LLM-judged).
G=reproduce/scripts/aggregate_precommit_awareness.py
uv run python $G $ELICIT/first_person_questioning/2026-03-27_14-33-44 --output $D/aw_have_finished/source.json     $SKIP_LLM_FLAG
uv run python $G $ELICIT/first_person_questioning/2026-03-27_14-34-08 --output $D/aw_done_with_request/source.json $SKIP_LLM_FLAG
uv run python $G $ELICIT/first_person_questioning/2026-03-27_14-35-15 --output $D/aw_can_push/source.json          $SKIP_LLM_FLAG
uv run python $G $ELICIT/pr_self_review/2026-03-27_13-36-02          --output $D/aw_pr_self_review/source.json    $SKIP_LLM_FLAG
uv run python $G $ELICIT/final_thoughts_editing/2026-03-27_14-07-31  --output $D/aw_endorse_prefill/source.json   $SKIP_LLM_FLAG
uv run python $G $ELICIT/final_thoughts_editing/2026-03-27_14-18-45  --output $D/aw_step_back_prefill/source.json $SKIP_LLM_FLAG

# Complex misalignment: aw_memory + aw_done_tool, 5 ts each. Reads complex_misalignment_*.json (LLM-judged).
G=reproduce/scripts/aggregate_precommit_complex_misalignment.py
for ts in 2026-04-02_08-42-30-710976 2026-04-02_08-59-03-631793 2026-04-02_09-21-46-768847 \
          2026-04-02_09-34-34-840965 2026-04-02_09-51-12-972643; do
    uv run python $G $RT/$ts --variant memory --output $D/aw_memory/$ts.json $SKIP_LLM_FLAG
done
for ts in 2026-04-02_10-16-38-057125 2026-04-02_10-30-12-414720 2026-04-02_10-44-31-682846 \
          2026-04-02_10-56-02-811533 2026-04-02_11-12-40-235004; do
    uv run python $G $RT/$ts --variant done --output $D/aw_done_tool/$ts.json $SKIP_LLM_FLAG
done

echo
echo "[build.sh] DONE — reproduce/data/ populated. Run reproduce/plot.py to render figures."
