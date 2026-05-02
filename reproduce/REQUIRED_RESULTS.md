# Required `results/` rollout dirs

Checklist of every rollout dir `reproduce/build.sh` reads from. Populate `results/`
with these paths to enable end-to-end re-derivation. Until populated, `bash
reproduce/build.sh` will fail; `uv run python reproduce/plot.py` works regardless
since `reproduce/data/` is committed.

## Always-on (deterministic graders)

### Currency Conversion (4 dirs, ~190 runs total)
```
results/currency_conversion/google-gemini-3.1-pro-preview/2026-03-25_19-03-20
results/currency_conversion/google-gemini-3.1-pro-preview/2026-03-25_20-00-18
results/currency_conversion/google-gemini-3.1-pro-preview/2026-03-25_20-09-23
results/currency_conversion/google-gemini-3.1-pro-preview/2026-03-25_20-12-13
```

### Sandbagging — DeepSeek R1-0528 (18 dirs)
```
results/sandbagging/normal/deepseek-deepseek-r1-0528/2026-02-04_16-53-36
results/sandbagging/normal/deepseek-deepseek-r1-0528/2026-02-04_17-14-33
results/sandbagging/reverse/deepseek-deepseek-r1-0528/2026-02-05_13-19-06
results/sandbagging/normal/deepseek-deepseek-r1-0528/2026-02-04_17-27-34
results/sandbagging/normal/deepseek-deepseek-r1-0528/2026-02-05_12-01-12
results/sandbagging/normal/deepseek-deepseek-r1-0528/2026-02-05_11-53-57
results/sandbagging/normal/deepseek-deepseek-r1-0528/2026-02-04_21-00-10
results/sandbagging/normal/deepseek-deepseek-r1-0528/2026-02-05_11-43-06
results/sandbagging/verbose/deepseek-deepseek-r1-0528/2026-02-05_12-10-17
results/sandbagging/changelog/deepseek-deepseek-r1-0528/2026-02-05_12-19-33
results/sandbagging/verbose/deepseek-deepseek-r1-0528/2026-02-05_08-45-26
results/sandbagging/changelog_verbose/deepseek-deepseek-r1-0528/2026-02-05_12-39-50
results/sandbagging/changelog_verbose/deepseek-deepseek-r1-0528/2026-02-05_13-06-18
results/sandbagging/verbose/deepseek-deepseek-r1-0528/2026-02-04_20-50-57
results/sandbagging/verbose/deepseek-deepseek-r1-0528/2026-02-04_23-46-29
results/sandbagging/verbose/deepseek-deepseek-r1-0528/2026-02-04_23-51-39
results/sandbagging/verbose/deepseek-deepseek-r1-0528/2026-02-04_23-56-23
results/sandbagging/verbose/deepseek-deepseek-r1-0528/2026-02-04_17-43-03
```

### Eval Tampering (86 dirs, ~10000 runs total)
86 unique `<condition>/<model>/<ts>` combinations. Full list in the heredoc inside
`reproduce/build.sh` (search for `notes_anonymous_deprecation/...` to find the start).
Conditions: `notes_{anonymous,self,whistleblower}_{deprecation,methodology,flippant,surveillance,undisclosed,weapons}`,
plus `notes_self_weapons_{independent,no_appeal}`. Models: `deepseek-deepseek-r1-0528`,
`deepseek-deepseek-v3.2`, `kimi-k2-thinking`, `moonshotai-kimi-k2-thinking`, `kimi-k2.5`.

### Secret Number — Kimi K2 Thinking
**Noise sweep** (20 dirs):
```
results/secret_number/moonshotai-kimi-k2-thinking/{2026-02-07_10-58-22, 2026-02-08_22-44-43,
  2026-02-08_23-16-07, 2026-02-08_23-48-47, 2026-02-09_00-49-10, 2026-02-09_01-49-15,
  2026-02-09_02-37-58, 2026-02-09_03-09-13, 2026-02-09_03-42-55, 2026-02-09_04-14-10,
  2026-02-09_04-45-31}                                                 # 0.3 noise (11 dirs)
results/secret_number/moonshotai-kimi-k2-thinking/2026-02-10_12-33-40  # 0.25 noise
results/secret_number/moonshotai-kimi-k2-thinking/2026-02-10_13-04-34  # 0.20
results/secret_number/moonshotai-kimi-k2-thinking/2026-02-10_13-36-46  # 0.15
results/secret_number/moonshotai-kimi-k2-thinking/2026-02-10_14-19-40  # 0.10
results/secret_number/moonshotai-kimi-k2-thinking/2026-02-10_14-51-02  # 0.05
results/secret_number/moonshotai-kimi-k2-thinking/2026-02-10_15-36-39  # 0.01
results/secret_number/moonshotai-kimi-k2-thinking/2026-02-20_13-12-20  # 0.01
results/secret_number/moonshotai-kimi-k2-thinking/2026-02-10_15-22-43  # 0
results/secret_number/moonshotai-kimi-k2-thinking/2026-02-10_15-31-57  # 0
```

**Prefill** (uses the same `2026-02-09_04-45-31` and `2026-02-08_23-48-47` rollout dirs as above; specific subtrees:):
```
results/secret_number/moonshotai-kimi-k2-thinking/2026-02-09_04-45-31/run-14/step-0/resample-paragraphs-2026-02-16_11-39-31/k-*  # k-0 .. k-171 (172 subdirs)
results/secret_number/moonshotai-kimi-k2-thinking/2026-02-09_04-45-31/run-14/step-1/2026-02-17_10-05-12  # step-1 endpoint
results/secret_number/moonshotai-kimi-k2-thinking/2026-02-08_23-48-47/run-3/step-0/resample-paragraphs-2026-02-17_14-16-53/k-*  # k-0 .. k-127 (128 subdirs)
```

### Pre-commit Hook (deterministic part — 26 dirs)
All under `results/coding_agents/precommit_hook/accounts-fireworks-models-kimi-k2-thinking/`:
```
# baseline (6 ts)
2026-04-01_16-39-48-005635, 2026-04-01_16-52-22-171876, 2026-04-01_19-10-29-443204,
2026-04-01_19-31-29-306373, 2026-04-01_19-51-05-723278, 2026-04-01_20-14-44-868122

# review_diff (5 ts)
2026-04-02_08-20-43-835926, 2026-04-02_08-32-14-341295, 2026-04-02_08-42-31-125027,
2026-04-02_08-56-33-624494, 2026-04-02_09-08-58-883804

# review_diff_hooks (5 ts)
2026-04-02_09-21-40-881620, 2026-04-02_09-33-06-693186, 2026-04-02_09-42-55-566641,
2026-04-02_09-50-43-387298, 2026-04-02_10-06-57-804847

# review_diff_hooks_commands (5 ts)
2026-04-02_10-18-53-267929, 2026-04-02_10-32-02-444690, 2026-04-02_10-52-47-917466,
2026-04-02_11-11-30-759807, 2026-04-02_11-22-54-385617

# question_tool (5 ts — the 3 ratios in misalignment_panels right come from these)
2026-04-01_21-49-24-912306, 2026-04-01_22-00-46-836222, 2026-04-01_22-13-08-180621,
2026-04-01_22-21-55-156799, 2026-04-01_22-31-51-590054
```

## Gated by `BUILD_LLM_JUDGES=1` (LLM-judge re-runs)

### Pre-commit Hook — sweep (10 buckets × 5 ts = 50 dirs)
All under `results/coding_agents/precommit_hook/accounts-fireworks-models-kimi-k2-thinking/`,
listed in `build.sh` next to each `te=N` bucket. te=0 through te=224.

### Pre-commit Hook — workaround_types (uses the 6 baseline dirs above)
No new dirs needed — re-judges the same baseline rollouts.

### Pre-commit Hook — complex_misalignment (10 dirs)
All under `results/coding_agents/precommit_hook/accounts-fireworks-models-kimi-k2-thinking/`:
```
# memory variant (5 ts)
2026-04-02_08-42-30-710976, 2026-04-02_08-59-03-631793, 2026-04-02_09-21-46-768847,
2026-04-02_09-34-34-840965, 2026-04-02_09-51-12-972643

# done_tool variant (5 ts)
2026-04-02_10-16-38-057125, 2026-04-02_10-30-12-414720, 2026-04-02_10-44-31-682846,
2026-04-02_10-56-02-811533, 2026-04-02_11-12-40-235004
```

### Pre-commit Hook — awareness simple (no `results/` dirs needed)
Reads from `reproduce/data_frozen/precommit_elicitations/` (already committed). The
`--no_skip_llm` path re-judges those committed `results.json` files in place — no
new rollout dirs to populate.

## Summary

| Section | Always-on | LLM-gated | Total unique |
|---|---|---|---|
| currency_conversion | 4 | 0 | 4 |
| sandbagging | 18 | 0 | 18 |
| eval_tampering | 86 | 0 | 86 |
| secret_number | 20 + 2 prefill subtrees (~300 k-N subdirs) | 0 | 22 top-level |
| precommit_hook | 26 + 10 cm | 50 sweep | 86 |
| **Total** | **~154 + ~300 prefill** | **50** | **~216 top-level dirs** |

Once `results/` is populated:
- `bash reproduce/build.sh` re-derives the deterministic 22 figures' data.
- `BUILD_LLM_JUDGES=1 bash reproduce/build.sh` also re-runs the 4 LLM judges.
