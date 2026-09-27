# Entity-resolution research pipeline

This project separates entity resolution into five replaceable stages:

1. preprocessing;
2. candidate retrieval;
3. pair matching;
4. decision policy; and
5. evaluation.

The leakage-safe data protocol, preprocessing interfaces, retrieval experiments,
and evaluation are implemented. The pair matcher and decision policy still have
explicit **Implementation TBD** boundaries. The next step is to implement those
two stages without changing the frozen splits, evaluation queries, or candidate
sets used for matcher comparisons.

Raw files under `dataset/` are read-only inputs. Historical EDA outputs under
`eda/` are preserved. New experiment outputs belong under the ignored
`artifacts/` directory.

Raw competition datasets and generated EDA data, reports, and notebooks are
intentionally excluded from Git. Place the supplied files under `dataset/train/`
and `dataset/test/` before running the pipeline.

## Architecture and contracts

| Module | Purpose | Main interface |
|---|---|---|
| `src/er/data.py` | Raw reads, frozen business groups, scoped label access, fingerprints | `SplitStore` and record iterators |
| `src/er/preprocess.py` | Derived representations while retaining original fields | `fit`, `transform`, `blocking_records` |
| `src/er/retrieve.py` | Full-pool indexing and label-free candidate retrieval | `build_index`, `retrieve`, blocking helpers |
| `src/er/match.py` | Score the exact retrieved pairs | `fit`, `score` |
| `src/er/decide.py` | Tune and apply an acceptance policy, including empty sets | `tune`, `apply` |
| `src/er/evaluate.py` | Stage and end-to-end metrics | `evaluate`, `summarize`, `blocking` |
| `src/er/run.py` | CLI orchestration and artifact permissions | `python -m er.run ...` |

Records retain `entity_id`, `country`, `business_name`, and
`business_address`. Source is encoded in the qualified entity ID. Candidate
pairs are uniquely keyed by `source1_entity_id` and `candidate_entity_id` and
carry deterministic ranks. Matcher outputs add `score`. Accepted IDs must be a
subset of the retrieved IDs, and every query remains present even when it has
zero candidates or an empty prediction.

Searchable Source 2/3 records are separate from labels available for fitting.
Validation and holdout labels are never training inputs. Evaluation uses the
candidate cache exactly as retrieved and never inserts missed true matches.

## Frozen protocol

`configs/experiments.toml` uses seed 42 and split version `v1`. Businesses are
grouped before assignment, stratified by country and match status, and assigned
approximately 80%/15%/5% to train, validation, and final holdout. Membership and
scoped labels are stored in `artifacts/splits/v1/`.

| Setting | Fit | Tune | Development feedback | Final check |
|---|---|---|---|---|
| Mixed | India + US train | India + US validation | Per-country validation results | India + US holdout |
| India → US | India train | India validation | US validation | US holdout |
| US → India | US train | US validation | India validation | India holdout |

Destination-country validation is development feedback and does not estimate
France performance. The final holdout remains unopened. `freeze` records a
completed design before `final` can access holdout labels, and final access is
ledgered even if evaluation is interrupted.

The official score is the mean per-business F0.5 from the supplied competition
definition. It includes empty predictions, no-match businesses, and retrieval
misses. Empty truth with an empty prediction scores 1; empty truth with a
nonempty prediction scores 0.

## Retrieval baseline and results

The retained full-pool BM25 index is
`eda/reports/full_pool/candidate_index.sqlite3`. It covers all 10,320,219
eligible Source 2/3 records, including distractors. At a fixed candidate budget
of 100, its saved deterministic prefix outperformed the evaluated character
TF-IDF route on every identical development query set, so BM25 is the retrieval
comparison point for the matching stage.

| Profile / role | Queries | TF-IDF ceiling | Micro recall | Complete recall | BM25 ceiling | TF-IDF delta |
|---|---:|---:|---:|---:|---:|---:|
| Mixed / tune | 40 | 0.971119 | 0.896000 | 0.775000 | 0.986452 | -0.015333 |
| India / tune | 20 | 0.955117 | 0.838235 | 0.650000 | 0.972904 | -0.017787 |
| US / destination | 20 | 0.987121 | 0.964912 | 0.900000 | 1.000000 | -0.012879 |
| US / tune | 20 | 0.987121 | 0.964912 | 0.900000 | 1.000000 | -0.012879 |
| India / destination | 20 | 0.955117 | 0.838235 | 0.650000 | 0.972904 | -0.017787 |

These are retrieval metrics and oracle ceilings, not achieved matcher or
end-to-end F0.5 scores. Every matched query had at least one true result, and no
query had zero candidates. The small fixed subsets happened to contain no
no-match businesses.

The completed TF-IDF index used 18,022,025,501 bytes and took 2,728 seconds to
build, with peak process RSS of 1,006,632,960 bytes. The dense pilot encoded
2,000 evenly sampled records in 46.0 seconds, used 2,428,485,632 bytes peak RSS,
and measured 0.9755 ANN-versus-exact Recall@100 on that reduced pool. A linear
full-pool estimate was about 66 CPU hours, so full dense retrieval was deferred.

Generated TF-IDF/dense indexes and the downloaded dense model were removed after
the reports and exact candidate caches were preserved. Rebuilding them creates
new content-addressed directories under `artifacts/indexes/`; the dense preflight
also downloads the pinned BGE-M3 revision into `artifacts/models/`.

## Retained artifacts

| Path | Reason retained |
|---|---|
| `artifacts/splits/v1/` | Frozen membership and scoped labels |
| `artifacts/evaluators/blocking-v2/` | Current independently frozen evaluator |
| `artifacts/evaluators/blocking-v1/` | Audit snapshot predating the validation fix |
| `artifacts/candidates/` | Exact fixed inputs for comparisons and historical baseline runs |
| `artifacts/runs/` | Configurations, metrics, per-query results, and error reports |
| `artifacts/blocking-execution.json` | Completion record for the blocking work |

Evaluator v2 prevents `missing_fields` metadata from overwriting authoritative
IDs, countries, counts, ranks, or scores. It accepts exactly the complete boolean
`missing_name` and `missing_address` fields. No quality report was produced with
the defective v1 evaluator.

Candidate caches are versioned by queries, raw inputs, preprocessing, retrieval
configuration/code, pool, and seed. Matcher comparisons must reuse the same
cache. Retrieval comparisons must report both coverage and downstream score.

## Commands

Set up the locked environment and run checks:

```sh
uv sync --locked
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test*.py'
PYTHONPATH=src .venv/bin/python -m unittest discover -s eda/analysis -p 'test*.py'
```

Prepare or reproduce the protocol baseline:

```sh
PYTHONPATH=src .venv/bin/python -m er.run prepare
PYTHONPATH=src .venv/bin/python -m er.run audit-legacy
PYTHONPATH=src .venv/bin/python -m er.run baseline --setting mixed
PYTHONPATH=src .venv/bin/python -m er.run baseline --setting india_to_us
PYTHONPATH=src .venv/bin/python -m er.run baseline --setting us_to_india
```

`prepare` refuses to replace an existing split directory. Existing splits should
be reused; changing the seed or membership rules requires a new split version.

Rebuild the deleted experimental retrieval assets only when resuming that work:

```sh
PYTHONPATH=src .venv/bin/python -m er.run blocking-preflight --branch lexical
PYTHONPATH=src .venv/bin/python -m er.run blocking-build --branch lexical
PYTHONPATH=src .venv/bin/python -m er.run blocking-preflight --branch dense
```

Blocking commands enforce a 16 GiB maximum, abort at 12 GiB measured RSS, use
disk-sharded indexes, and serialize concurrent project jobs with a lock.

After choosing a completed development run, final evaluation is explicit:

```sh
PYTHONPATH=src .venv/bin/python -m er.run freeze --run artifacts/runs/RUN_ID
PYTHONPATH=src .venv/bin/python -m er.run final --run artifacts/runs/RUN_ID
```

Do not run `final` during development.

## Matching-stage handoff

The next implementation must fill the existing `match.py` and `decide.py`
interfaces. Architecture, features, losses, negative sampling, calibration,
thresholds, and decision rules remain **Implementation TBD**.

Matcher training must use candidates retrieved for training queries, including
realistic hard negatives. Comparisons must keep candidate pairs fixed, tune only
on the setting's authorized validation group, report matching runtime and
retrieved-pair precision/recall, and preserve empty predictions. End-to-end
reports must continue to use the exact official macro F0.5 and classify errors as
retrieval misses, matcher false rejections, or false acceptances.
