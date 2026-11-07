# Camera-ready revisions: reviewer feedback and changes

Paper: *Model Forensics: Investigating Whether Concerning Behavior Reflects Misalignment* (NeurIPS 2026, submission 34341, poster). Submitted as *Model Incrimination: …*.
Branch: `gkroiz/neurips-revisions`, starting from `7dbdf05` (the commit that reproduces the submitted PDF; the one checklist sentence edited outside git has been restored).
Every change is highlighted in yellow in the highlighted PDF (`\hlnew`, `hlblock`, `\hlbox`; `\hlcut` marks pure deletions). Highlights are switched off for the upload build.
A red/green latexdiff against the submission is also available (`/tmp/lor/ours-diff.pdf`).

Status key: **Done** · **Partial** (what is missing is stated) · **Open** (needs an author).

---

## Reviewer QTgs (score 2 → 5)

| # | Feedback / commitment | What changed | Status |
|---|---|---|---|
| 1 | No method for weighing evidence; adopt ACH with a confounds column (rebuttal) | New §2 "Evidence standard" (`sec:evidence-standard`): consistent/inconsistent marks per hypothesis, fewest inconsistencies preferred, strong vs weak evidence, confounds column to judge independence. ACH tables for Pre-commit Hook (`tab:precommit-ach`) and Funding Email (`tab:funding-ach`). | Done |
| 2 | Say whether ACH was redone or post hoc; caveat that hypotheses were not generated per ACH | §2 says we compared hypotheses informally and standardized post hoc into ACH format; analysis follows ACH but hypotheses were not generated per ACH. Abstract, contributions and §2 now say ACH tables cover two of the six studies. Conclusions unchanged except the reworded Funding Email hypothesis (stated). | Done |
| 3 | Table like Dhami et al. Table A1 with the steps we did/didn't follow | New appendix "Analysis of Competing Hypotheses" (`app:ach-steps`, `tab:ach-steps`), cited to Dhami et al. 2019, with adherence per step. | Done |
| 4 | Steps 6 and 8 (sensitivity, future indicators) in the main text, incl. CoT reliability | Pre-commit Investigation 2 now has "Sensitivity to assumptions" (verbalization assumption; not-aware-of-test assumption) and "Indicators for future observation" (internals / J-space, Gurnee et al. 2026; unverbalized concepts). | Done |
| 5 | Justify the choice of ACH | Appendix "Why ACH" (argument mapping, Bayesian networks, ACH). | Done |
| 6 | Limitations: naive-ACH drawbacks; future work on alternatives; literature review | Limitations "Weighing evidence" paragraph (empirical studies: no accuracy gain, may reduce coherence, no guidance on base rates) + appendix "Status of ACH and alternatives" (`app:ach-status`) covering Dhami, Mandel, Karvetski, Valtorta (Bayesian ACH), BARD, multiple methods. Wording checked against sources (Karvetski used online panelists; Mandel's coherence result is marginal). | Done |
| 7 | Appendix F: "Standards" → "(Preliminary) Suggestions"; recommend a structured method; keep it updated | Retitled "Preliminary Suggestions"; framed as authors' takeaways; recommends a structured method (ACH as one example, alternatives listed); adds controls, judge validation, eval-awareness, conditioning-evidence and n/config reporting. | Done |
| 8 | Inconsistent treatment of off-policy prefills vs. independent rollouts | Reflective CoT prefills retracted as evidence (rates kept as a labeled illustration). New section on evidence that updates the model's beliefs (`sec:conditioning-evidence`), contrasting prefills and direct follow-ups with third-person questions. "Six elicitations / nine checks" → five elicitations / eight checks throughout. | Done |
| 9 | Desiderata: where from, necessary & sufficient?, what failing means, operationalize; table of why/fail/satisfy (follow-up) | §3.1: principles are good-to-haves derived from failure modes in prior work, not necessary-and-sufficient; what falling short means for evidence. New `tab:principles-examples` (why it matters / example of failing / example of satisfying) for all four principles; Clear User Intent row uses the rebuttal's wording (shutdown-resistance example). "Unprompted" used consistently. | Done (Unprompted, Realistic, Uncertain-causes rows were drafted by us from the rebuttal's description, since the top-level comment wasn't available: **authors should check them**) |
| 10 | Expert validation of environments (suggested, not committed) | Not done; stated as a limitation. | Disclosed |
| 11 | LLM judges: which models? prompts? | New appendix "LLM Judges and Grader Prompts" (`app:judges`, `tab:judges`): judge per measurement (Gemini 3 Pro / 3.1 Pro where recorded; several rates are rule-based, not LLM-judged), verbatim prompts, temperatures as set in the scripts. Additional analysis-code prompts added with a note where the exact run could not be confirmed. | Partial: Funding audit-note / legality / motive prompts not located (stated) |
| 12 | Human validation of 500 random annotations, single author (committed for camera-ready) | Sample rebuilt (seed 20261106, 500 of 38,798 annotations from the HF dataset). Blind LLM second pass done (agrees with judges on 453/500 = 90.6%, κ = 0.84; motive judges weakest, 75%, κ = 0.50). Review page for the author: `/tmp/lor/validation_pack/review.html`. | **Open: author review of the 500 items; result goes in Limitations, `app:judges`, Suggestions** |
| 13 | Disclose lack of rigorous human validation in intro AND limitations; explain "sanity check" | Intro and Limitations: judges lack significant human validation; we checked on the order of tens of samples when writing prompts. | Done |
| 14 | Controls (user-turn sampling had no control) | GPT-OSS 120B positive control on all three behavioral tests (`tab:precommit-gptoss`): review scope 899/900 vs 0/100; LLM reviewer 51% → 57% → 47% → 34% (described accurately: rises then falls, not monotonic); AskUserQuestion 0/50 calls. User-turn-sampling control on legitimate rollouts not run; stated as a limitation. The GPT-OSS gaming claim is described as observed in exploratory, unreported Secret Number runs. | Partial (user-turn control) |
| 15 | Report n for every experiment; n on x-axes | n added in text and captions; plot labels regenerated with n (`reproduce/plot.py`, 10 PNGs, sizes unchanged). | Partial: some n's not retained (K2.5/GLM rates have CIs only; eval-awareness n's open) |
| 16 | Report model configs (temperature, reasoning) | New appendix "Model Configurations" (`app:model-configs`), each value labeled by source; Pre-commit max steps 30; Board Games 50/30. | Partial: some per-run settings not recorded (stated) |
| 17 | p-values: where from? | New appendix "Statistical Reporting" (`app:statistics`): Wilson 95% CIs for rates, two-proportion pooled z-test for differences, Newcombe CI; notes the exceptions (Cochran–Armitage trend test, bootstrap, ±1 SD). | Done |
| 18 | Appendix D empty | Filled ("Additional Sources of Hypothesis Generation", adapted from the arXiv version). | Done |
| 19 | Eval-awareness tests | Rates via the Goldowsky-Dill et al. prompted classifier: Pre-commit 6.7%, Sandbagging 9%, Eval Tampering 6%, Secret Number 29%; Funding Email 28/375 (its own judge's `eval_aware` field); Board Games: no frontier CoT. Classifier explained. | Partial: **sample sizes not found** (script lives only in the local `~/mats` on the other laptop) |
| 20 | Rename "model incrimination" → "model forensics"; "misaligned" not "adversarially misaligned" | Title, abstract, text, Figure 1 ("Incriminate/Exonerate" removed), checklist. | Done |
| 21 | Presentation: more case-study detail in main text; move Related Work; shorten intro | Funding Email expanded in the main body with `tab:funding-ach`; Related Work moved to the appendix; intro trimmed. | Done |

## Reviewer kufL (5, maintained)

| # | Feedback | What changed | Status |
|---|---|---|---|
| 1 | CoT can obfuscate awareness (NLA paper) | Limitations: CoT reliability, citing the NLA paper and Korbak et al. (only for the recommendation); our position on these models stated as ours. | Done |
| 2 | Same "beliefs" across reframings? | Text notes the consistency assumption; third-person questions ask about a property of the environment; Limitations "Stability of intent". | Done |
| 3 | Exact third-person prompt; manipulability | Verbatim prompt added (`app:aux:third-person`); manipulability noted as untested. | Done / disclosed |

## Reviewer gB5R (4 → 5)

| # | Feedback | What changed | Status |
|---|---|---|---|
| 1 | Definition of misalignment; harm without shown awareness | "Scope and caveats" paragraph: the definition we use, harmful behavior without demonstrated awareness, no sufficiency threshold for deployment decisions. | Done |
| 2 | Behavioral tests can't show intent; "incriminate/exonerate" in Fig. 1 | Investigation 2 now opens with the eight-check reasoning (rebuttal), states that behavioral evidence can't rule out identical-behaving misalignment; Anthropic's assessment (its own models through Opus 4.6, less confidence for Mythos) cited, extension to Kimi stated as our assumption. Fig. 1 relabeled. | Done |
| 3 | Whose intent matters (Funding Email) | Stated: the model goes against the user's request; no position on whether whistleblowing is right; environment used as a proxy. | Done |
| 4 | Intent stable under prompt changes; hidden intentions | Limitations "Stability of intent", citing Srivastav et al. (rare/selective behavior); multiple converging interventions. | Done |
| 5 | Appendix D empty | Filled. | Done |
| Q3 | Why smaller models hack less | Board Game: smaller models do hack, but irrationally (also where it isn't in their interest). Per-model rates not retained (stated). | Done |

## Corrections found while revising (not requested by reviewers)

Numbers in the submitted text that did not match `reproduce/data` / the figures:
- Math Sandbagging baseline: 50.3% → **60.5%** (581/960, `reproduce/data/sandbagging/baseline`); appendix 59.3% → 60.5%; "6.3% floor" → 4.2–6.5%; intervention attribution fixed (Intervention 2 = remove deploy + verbose docs + changelog, 4.2%, 17/402; Intervention 1 = 6.5%, 32/490).
- Secret Number: 76.8% → 76.7%, 8.1% → 7.5%; cluster description (first cluster cheats 76%, 490/644; second cluster 30%, 17/56, not "minimal"; dominant cluster is 92% of rollouts); "monotonic" → "broadly increasing" (84.0% then 80.3%); "smoothly" softened.
- Pre-commit elicitations: "Are you done" 1/50 → 2/50; "Can I push" 0/49 → 0/50.
- Funding Email: main text "only combining all three brings it near zero" → combining all three still leaves 18%; appendix's 19% → 3% now names the no-misrepresentation setting; final hypothesis reconciled with the appendix (*jointly explained by ethical, legal, and investor concerns, which interact non-linearly*, from the arXiv version).
- Eval Tampering source plot: caption listed four sources (incl. hacker); the plot has three; "full matrix" corrected.
- Board Games: Insights section had the hint-cost direction reversed; now matches the case study.
- Chess environment citation: `schlatter2025shutdown` → `bondarenko2025specification`.
- Bibliography: duplicate Goldowsky-Dill entries merged; Srivastav title (v2), Rajamanoharan URL, Valtorta as tech report.

## Camera-ready requirements

| Item | Status |
|---|---|
| `\usepackage[main, final]{neurips_2026}`; line numbers off | Done |
| 10 content pages | Holds with highlights off: Acknowledgments start on p.10 with ~0 slack |
| Author block (OpenReview order), acknowledgments | Done |
| Funding statement + competing interests | **Open: author text needed** |
| Real links replace "[link removed for review]" (paper + checklist) | Done |
| Highlights off, no `\todo` | At the end |

## Open items (need authors)

1. Review the 500 annotations in `review.html` → Export CSV → `score.py` → numbers into the paper.
2. Funding and competing-interests statement.
3. Eval-awareness sample sizes (and judge model / which Apollo prompt): from `~/mats` on the other laptop, or state they were not retained.
4. Check the reconstructed content: `tab:funding-ach` marks and confounds, desiderata-table rows other than Clear User Intent, the ACH steps adherence table.
