# Entity-resolution EDA: final findings

EDA is complete. This folder contains the findings, executed notebooks, reproducible analysis code, sampled data, plots, retrieval index, and matching experiments. Original dataset files remain unchanged in the project's `dataset/` directory.

The main finding is that the full-pool shortlist contains most labeled matches, while the baseline matcher rejects many of them. Cleanup is useful, but its measured gains are modest at a fixed acceptance threshold. The larger recall gain from combined cleanup includes a substantial threshold effect. For the current recall-focused stage, keep that tradeoff visible and evaluate how many true matches survive each stage.

## Evidence and scope

- Initial EDA: 2,000 training Source 1 references sampled with seed 42, proportionally across country and singleton status; all 6,933 labeled matches; 25,000 additional candidate distractors; 25 reviewed examples.
- Full-pool experiment: the same references searched against all 10,320,219 training Source 2/3 records. Independent name and address routes returned up to 100 candidates each, yielding 393,703 deduplicated pairs.
- Matching baseline: grouped reference splits of 1,200 training, 400 validation, and 400 held-out references. Matching labels, identical canonical reference names, and identical nonempty reference addresses were used to keep related references together.
- Cleanup experiments: the existing train/validation splits and 314,766 candidate pairs were reused. The classifier and feature list stayed fixed. No retrieval index rebuild or cleanup test-fold evaluation was performed.

All reported experiments use supplied training data. The held-out fold below is a split of training references, not the competition test dataset. Validation was reused for threshold selection and cleanup comparisons, so its small score differences are exploratory.

## What the data tells us

**Repeated names do not establish branches.** A targeted scan found another reference with the same name after common legal tokens were removed for 922/2,000 sampled references (46.1%). Names clearly collide across labeled entities and locations. Corporate branch relationships remain unresolved; name equality alone is insufficient for merging.

**Numeric differences need interpretation.** Among 6,012/6,933 true links with numeric address tokens on both sides, 505/6,012 (8.4%) had disjoint raw digit-token sets. Leading-zero normalization reduced this to 336/6,012 (5.6%). These are untyped numbers: shared postcodes can conceal conflicting building numbers, and added unit/floor numbers can create harmless differences. Preserve distinctions such as `123`, `12-3`, and `12/3`, but do not treat every numeric disagreement as a mismatch.

**Abbreviations explain only part of the name variation.** Exact name agreement rose from 322/6,933 (4.6%) raw to 1,554/6,933 (22.4%) after basic normalization, then to 1,744/6,933 (25.2%) after legal-name expansion. However, 665/6,933 true links (9.6%) still had name similarity below 0.3. Examples include different scripts and substantially different names at the same address. A changed trading name is a possible explanation, not something the labels prove.

**Strong similarity can mislead.** Reviewed nonmatches include identical names at different addresses and businesses sharing a building or landmark. The name-versus-address plots show useful patterns, but neither field alone reliably determines identity. Original hard negatives were selected from the small sampled pool using the plotted similarities; the later retrieved negatives come from the full candidate pool. Neither negative distribution estimates production prevalence.

**Missing fields and country differences matter.** Candidate addresses are blank in 344,883/10,320,219 full training records (3.34%), and in 299/6,933 sampled true links (4.31%). Training covers India and the US; competition test data also includes 259,452 France reference entities. Training labels cannot directly establish French matching accuracy. Missing-address handling and cross-script matching were deliberately left outside the cleanup work.

## Retrieval and matching findings

The top-100-per-route union retrieved **6,677/6,933 true links (96.31%)**. It recovered every labeled match for **1,698/1,888 nonsingleton references (89.94%)**, with an ideal-matcher macro F₀.₅ ceiling of **0.9844 across 2,000 references**. Known positives were never inserted into retrieval results.

The held-out baseline matcher scored **0.8516 macro F₀.₅ across 400 references**, with precision **1,012/1,041 (97.21%)** and recall **1,012/1,381 (73.28%)**. Its errors were:

| Error type | True links or accepted negative pairs |
|---|---:|
| True links absent from retrieval | 57 |
| Retrieved true links rejected by the matcher | 312 |
| Accepted nonmatches | 29 |

This points to matcher rejection as the larger current source of lost recall. Retrieval still has gaps, particularly in the India, missing-address, and low-name-similarity subgroups; detailed denominators are in the saved metrics.

Singletons have no matches to retrieve. Their relevant measures are misleading candidates and false acceptance. All 112 sampled singletons received candidates; the held-out matcher falsely accepted a candidate for 4/23 singleton references (17.39%). This small denominator limits interpretation.

## Cleanup: what changed and whether it helped

The cleanup preserves script letters and combining marks, case-folds text, removes appended websites while retaining stems for website-only names, joins dotted legal suffixes, normalizes leading zeros, and retains numeric separators. It operates on derived values; original names and addresses remain available.

Compared with the baseline representation, combined cleanup changed at least one derived field for **227/1,600 references (14.19%)** and **53,096/292,353 unique retrieved candidate records (18.16%)** in the train/validation experiment. These are unique-record counts, not repeated pair counts or full-corpus estimates.

| Individual rule | Changed references / 1,600 | Changed unique candidates / 292,353 |
|---|---:|---:|
| URL removal and domain-stem fallback | 0 (0.00%) | 10,809 (3.70%) |
| Dotted legal suffixes | 39 (2.44%) | 3,081 (1.05%) |
| Leading-zero cleanup | 19 (1.19%) | 11,957 (4.09%) |
| Numeric-separator preservation | 174 (10.88%) | 30,668 (10.49%) |

Rule counts overlap and cannot be summed. The detailed report also counts every processing step separately for names and addresses. Lowercasing and general punctuation cleanup were already present in the baseline.

The following results cover the same **400 validation references and 1,383 true links**, including unretrieved matches in the recall denominator. Every single-rule run selected threshold 0.99; combined cleanup selected 0.98. Thresholds were selected on macro F₀.₅, which favors precision—not explicitly optimized for the newly emphasized recall objective.

| Experiment | Threshold | Macro F₀.₅ | Precision: TP / accepted pairs | Recall: TP / 1,383 |
|---|---:|---:|---:|---:|
| Baseline | 0.99 | 0.8651 | 1,027/1,063 (96.61%) | 74.26% |
| URL only | 0.99 | 0.8679 | 1,041/1,082 (96.21%) | 75.27% |
| Dotted suffix only | 0.99 | 0.8693 | 1,043/1,080 (96.57%) | 75.42% |
| Leading zeros only | 0.99 | 0.8676 | 1,041/1,078 (96.57%) | 75.27% |
| Numeric separators only | 0.99 | 0.8686 | 1,034/1,067 (96.91%) | 74.77% |
| Combined cleanup | 0.98 | 0.8677 | 1,138/1,227 (92.75%) | 82.28% |
| Same combined-model scores at 0.99 | 0.99 | 0.8642 | 1,033/1,070 (96.54%) | 74.69% |

**For recall, combined cleanup at its selected threshold accepted 111 more true links net than the baseline**, with 53 more accepted nonmatches net. This is a promising operating point for an intermediate filter, but it does not prove that cleanup alone supplies an eight-percentage-point recall improvement. On the same combined-model scores, moving from 0.98 to 0.99 removes 83 of the 114 newly recovered positives and 45 of the 54 newly accepted negatives. A comparison at equal candidate volume or another agreed operating constraint remains necessary to choose a recall-oriented setting fairly.

Single-rule comparisons show more limited changes on pairs whose text actually changed:

| Rule | Affected validation positives / negatives | Recovered / newly rejected positives | Corrected / new false acceptances |
|---|---:|---:|---:|
| URL | 64 / 2,722 | 2 / 2 | 0 / 0 |
| Dotted suffix | 50 / 2,445 | 4 / 1 | 0 / 1 |
| Leading zeros | 57 / 3,973 | 1 / 0 | 0 / 0 |
| Numeric separators | 213 / 13,825 | 4 / 4 | 6 / 0 |

Dotted suffix normalization had the strongest overall single-rule macro-score result. Numeric separators offered the clearest reduction in false acceptance on affected pairs. Leading zeros had a small positive result. URL removal was mixed: it helped names such as `wardelser.com`, but `Mr lbrothers.com → mr` loses informative text because the qualifier prevents the website-only fallback. That defect is documented and remains unfixed.

Retraining also changed decisions on unaffected text. Results from different rules overlap, so recoveries and mistakes cannot be assigned exclusively to one rule or added together. Cleanup's effect on full-pool retrieval remains unmeasured because the original shortlist was reused.

## Conclusions for the next stage

1. **Prioritize retaining true matches through the matcher.** Its rejected retrieved positives are the largest measured recall loss. Compare recall against retained candidate volume and false acceptance using the existing validation scores; do not assume 0.98 is already the best recall-focused threshold.
2. **Carry forward the versioned cleanup candidates, with their evidence and limitations.** Dotted suffixes and numeric structure are useful; URL removal needs the documented qualifier case addressed before treating it as reliable. No experiment has established that combining all four rules is superior at an equal operating constraint.
3. **Confirm the selected pipeline on an appropriate holdout.** Cleanup comparisons reused validation, singleton counts are small, and retrieval gains from cleanup remain unknown. Missing-address and cross-script work belong to the subsequent matching/retrieval phase, not an extension of this completed cleanup EDA.

## Artifact guide and reproduction

| Location | Contents |
|---|---|
| [Detailed findings](reports/eda_first_pass.md) | Original EDA, retrieval results, per-step cleanup counts, ablations, and examples |
| [EDA notebook](notebooks/entity_resolution_eda.ipynb) | Executed first-pass statistics, normalization, example review, and similarity plots |
| [Retrieval and baseline notebook](notebooks/full_pool_retrieval_baseline.ipynb) | Executed full-pool retrieval, matcher, cleanup, and ablation comparisons |
| [Analysis code](analysis/) | Reproducible runners, tests, initial inventory, and review notes |
| [Sampled EDA outputs](reports/eda_data/) | References, labels, sampled records and pairs, 25-example casebook, and statistics |
| [Full-pool outputs](reports/full_pool/) | SQLite index, candidates, metrics/configuration, baseline predictions, errors, and plot |

The folder is about 2.9 GiB, dominated by the reusable SQLite index. Artifacts were moved, not duplicated. Raw data and the shared project environment (`pyproject.toml`, `uv.lock`, `.venv`) remain at the project root. This folder depends on those for reruns.

Run these checks from the project root:

```sh
uv run python -m unittest discover -s eda/analysis -p 'test_*.py'
uv run python eda/analysis/full_pool_baseline.py --validate-only
```

The relocated notebooks find the project root from their working directory and load the saved results by default. Both retain their executed outputs. Explicit experiment reruns are available through `eda/analysis/full_pool_baseline.py --cleanup-comparison` and `--cleanup-ablations`; the default runner reruns retrieval and matching. The initial EDA runner is `eda/analysis/eda_first_pass.py`, with `--refresh` to rebuild the sample. Reruns write into this folder and can replace derived results.

Raw input size/mtime fingerprints were verified against the original inventory. The index, candidate table, and experiment metrics retained their size/mtime fingerprints during organization. The move did not retrain models or alter saved predictions. Tests and relocated notebook execution validate that the consolidated paths remain usable.
