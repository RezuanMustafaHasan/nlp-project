# Exploratory reused-test results

Exploratory evaluation on the reused retained benchmark; no untouched-test claim

Independent decoding; all candidates frozen using validation. Full retained test coverage, identical scoring contract.

| Candidate | Macro-F1 | Micro-F1 | Ordered exact match | Span-F1 | False-paren sentences / 1,000 negatives |
|---|---:|---:|---:|---:|---:|
| compact_joint_initial_s42 | 46.63 | 89.41 | 77.26 | 38.32 | 1.152 |
| compact_ce_s42 | 47.76 | 89.82 | 78.00 | 40.39 | 1.081 |
| compact_weighted_s42 | 48.63 | 89.00 | 76.39 | 41.13 | 8.351 |
| banglabert_initial_s42 | 64.03 | 90.38 | 77.71 | 72.36 | 2.314 |
| banglabert_aligned_weighted_s42 | 68.93 | 92.96 | 83.46 | 74.89 | 1.687 |
| banglabert_historical_s42 | 65.01 | 93.23 | 84.20 | 72.37 | 1.392 |
| bilstm_initial_s42 | 50.65 | 90.30 | 78.80 | 43.31 | 1.386 |
| bilstm_packed_weighted_s42 | 52.05 | 89.77 | 77.93 | 43.01 | 9.088 |
| bilstm_historical_s42 | 51.13 | 90.31 | 78.81 | 43.09 | 1.386 |
| compact_joint_initial_s123 | 46.05 | 89.45 | 77.27 | 37.65 | 0.802 |
| compact_selected_s123 | 47.68 | 89.83 | 77.96 | 40.37 | 0.933 |
| compact_joint_initial_s2026 | 45.61 | 89.26 | 76.98 | 35.84 | 0.748 |
| compact_selected_s2026 | 46.81 | 89.66 | 77.66 | 37.15 | 0.688 |

## Compact continuation: three seeds

Sample SD measures variation in observed gains across these three seeds. The percentile interval resamples paired sentences, conditional on the fixed checkpoints.

- macro_f1_pp: +1.316; sample SD 0.269; conditional 95% interval [+1.133, +1.497].
- ordered_exact_match_pp: +0.707; sample SD 0.033; conditional 95% interval [+0.654, +0.757].
- span_f1_pp: +2.031; sample SD 0.702; conditional 95% interval [+1.417, +2.721].

Per-symbol counts and positive/negative partition scores are in each metrics JSON; paired intervals are in paired_intervals.json. Historical baseline inference is labeled separately from corrected inference. The corrected baselines' continuation changes weighting and exposure together. No UI default or historical manuscript is changed by this evaluation.
